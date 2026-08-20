"""
Exp-11: SOTA Best-of-N Precision Hybrid with CLIP MMR Reranker
-------------------------------------------------------------
1. High-Rank LoRA (Exp-05 / Exp-08) + 2nd-Order Heun Controlled Inversion (50 Steps)
2. Clean Foreground Reference (_nobg.png) to eliminate background noise
3. Over-generation: 4 candidates per prompt with latent perturbation
4. CLIP Multi-Objective Selection:
   Score = W_T * CLIP-T + W_I * CLIP-I - W_DIV * max_sim(prev_picks) - DUP_PENALTY * (dup - DUP_TH)
5. Official Evaluation & Extended Suite Verification
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel

# Reuse proven schedulers and utils from generate_hybrid.py
from generate_hybrid import (
    CLASS_PROMPT,
    DEFAULT_NEGATIVE_PROMPTS,
    ControlledHeunODE,
    ControlledODEInversion,
    get_reference_latent,
)

SUBJECT_ROUTING = {
    "actionfigure_2":       {"tau": 0.72, "eta": 0.88, "ref_mode": "nobg", "w_i": 1.2, "w_t": 1.0},
    "decoritems_woodenpot": {"tau": 0.80, "eta": 0.92, "ref_mode": "nobg", "w_i": 1.2, "w_t": 1.0},
    "furniture_sofa2":      {"tau": 0.80, "eta": 0.92, "ref_mode": "nobg", "w_i": 1.2, "w_t": 1.0},
    "instrument_music2":    {"tau": 0.75, "eta": 0.88, "ref_mode": "nobg", "w_i": 1.2, "w_t": 1.0},
    "luggage_backpack1":    {"tau": 0.80, "eta": 0.90, "ref_mode": "nobg", "w_i": 1.2, "w_t": 1.0},
    "person_3":             {"tau": 0.72, "eta": 0.88, "ref_mode": "nobg", "w_i": 1.3, "w_t": 1.0},
    "pet_cat5":             {"tau": 0.78, "eta": 0.90, "ref_mode": "nobg", "w_i": 1.2, "w_t": 1.0},
    "scene_waterfall":      {"tau": 0.80, "eta": 0.92, "ref_mode": "first", "w_i": 1.2, "w_t": 1.0},
    "transport_tank":       {"tau": 0.72, "eta": 0.88, "ref_mode": "nobg", "w_i": 1.2, "w_t": 1.0},
    "wearable_jacket1":     {"tau": 0.75, "eta": 0.88, "ref_mode": "nobg", "w_i": 1.2, "w_t": 1.0},
}


def _unwrap(x):
    return x if isinstance(x, torch.Tensor) else x.pooler_output


@torch.no_grad()
def clip_text_emb(clip_model, clip_proc, prompts: List[str], device: str = "cuda") -> torch.Tensor:
    b = clip_proc(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    return _unwrap(clip_model.get_text_features(**b)).float()


@torch.no_grad()
def clip_image_emb(clip_model, clip_proc, images: List[Image.Image], device: str = "cuda", bs: int = 16) -> torch.Tensor:
    out = []
    for i in range(0, len(images), bs):
        b = clip_proc(images=images[i:i + bs], return_tensors="pt").to(device)
        out.append(_unwrap(clip_model.get_image_features(**b)).float())
    return torch.cat(out)


def blend_anchor(anchor: torch.Tensor, seed: int, strength: float = 0.20, device: str = "cuda") -> torch.Tensor:
    if strength <= 0.0:
        return anchor.clone()
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(anchor.shape, generator=g, device=anchor.device, dtype=anchor.dtype)
    return (1.0 - strength) * anchor + strength * noise


def main():
    parser = argparse.ArgumentParser(description="Exp-11: Best-of-N Precision Ensemble")
    parser.add_argument("--root", type=str, default="/content/project-3")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/exp05_lora_hq")
    parser.add_argument("--output_dir", type=str, default="./experiments/11_best_of_n_ensemble")
    parser.add_argument("--candidates", type=int, default=4, help="후보 생성 수 (프롬프트당)")
    parser.add_argument("--steps", type=int, default=50, help="Heun 적분 스텝 수")
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("=" * 80, flush=True)
    print("🏆 [Exp-11] SOTA Best-of-N Precision Ensemble with CLIP MMR Selection", flush=True)
    print(f"• LoRA Base Checkpoints: {args.checkpoints_dir}", flush=True)
    print(f"• Output Directory: {args.output_dir}", flush=True)
    print(f"• Candidates per Prompt: N = {args.candidates} (총 400장 생성 후 최적 100장 선별)", flush=True)
    print(f"• Solver: 2nd-Order Heun Predictor-Corrector ({args.steps} Steps)", flush=True)
    print("=" * 80, flush=True)

    # 1. CLIP 모델 로드
    clip_id = "openai/clip-vit-base-patch32"
    print(f"📦 CLIP 채점 모델 로딩 중: {clip_id}...", flush=True)
    clip_model = CLIPModel.from_pretrained(clip_id).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_id)

    # 2. SD3.5 파이프라인 로드
    print("📦 SD3.5-medium 파이프라인 로딩 중...", flush=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    base_sched_cfg = dict(pipe.scheduler.config)

    cands_dir = os.path.join(args.output_dir, "candidates")
    os.makedirs(cands_dir, exist_ok=True)

    t0_all = time.time()

    for concept, class_noun in CLASS_PROMPT.items():
        print(f"\n{'='*70}\n▶ [{concept}] Processing (Class: '{class_noun}')...\n{'='*70}", flush=True)
        cfg = SUBJECT_ROUTING.get(concept, {"tau": 0.72, "eta": 0.88, "ref_mode": "nobg", "w_i": 1.2, "w_t": 1.0})

        # A. LoRA 가중치 로드
        lora_path = os.path.join(args.checkpoints_dir, f"lora_{concept}")
        if not os.path.exists(lora_path):
            lora_path = os.path.join(args.root, "checkpoints", "exp08_dreambooth_lora", f"lora_{concept}")
        print(f"  🔗 LoRA 가중치 주입: {lora_path}", flush=True)
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path, torch_dtype=dtype)

        # B. Reference Latent & Inversion Trajectory 계산
        ref_latent = get_reference_latent(
            pipe=pipe,
            concept=concept,
            dataset_dir=os.path.join(args.root, "dataset"),
            aug_dir=os.path.join(args.root, "augmentation"),
            ref_mode=cfg["ref_mode"],
            seed=args.seed
        ).to(device=device, dtype=dtype)

        print(f"  📸 Controlled ODE Inversion (Heun {args.steps} Steps)...", flush=True)
        inversion_scheduler = ControlledODEInversion.from_config(base_sched_cfg)
        inversion_scheduler.set(
            reference=torch.randn_like(ref_latent, generator=torch.Generator(device=device).manual_seed(args.seed)),
            tau=0.0,
            eta=0.5
        )
        pipe.scheduler = inversion_scheduler

        with torch.no_grad():
            inv_out = pipe(
                prompt="",
                guidance_scale=1.0,
                num_inference_steps=args.steps,
                output_type="latent",
                latents=ref_latent,
            )
            inverted_latent = inv_out.images.clone()

        # C. 프롬프트 로드
        prompt_file = os.path.join(args.root, "prompt", f"{concept}.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            raw_prompts = [l.strip() for l in f.readlines() if l.strip()]

        eval_prompts = [p.replace("{}", class_noun) for p in raw_prompts]
        gen_prompts = [p.replace("{}", f"sks {class_noun}") for p in raw_prompts]
        neg_prompt = DEFAULT_NEGATIVE_PROMPTS.get(concept, "blurry, low quality, distorted, bad anatomy")

        concept_cands_dir = os.path.join(cands_dir, concept)
        os.makedirs(concept_cands_dir, exist_ok=True)
        concept_out_dir = os.path.join(args.output_dir, concept)
        os.makedirs(concept_out_dir, exist_ok=True)

        all_candidates = {pi: [] for pi in range(len(raw_prompts))}

        # D. 후보 생성 (N=4 candidates per prompt)
        print(f"  ⚡ 10개 프롬프트 × {args.candidates}개 후보 이미지 생성 중...", flush=True)
        for pi, (gen_p, eval_p) in enumerate(zip(gen_prompts, eval_prompts)):
            for ci in range(args.candidates):
                cand_seed = args.seed + pi * 1000 + ci * 37
                cand_lat = blend_anchor(inverted_latent, cand_seed, strength=0.18 * ci, device=device)

                gen_scheduler = ControlledHeunODE.from_config(base_sched_cfg)
                gen_scheduler.set(
                    reference=ref_latent,
                    tau=cfg["tau"],
                    eta=cfg["eta"],
                    schedule="adaptive"
                )
                pipe.scheduler = gen_scheduler

                generator = torch.Generator(device=device).manual_seed(cand_seed)

                with torch.no_grad():
                    img = pipe(
                        prompt=gen_p,
                        negative_prompt=neg_prompt,
                        num_inference_steps=args.steps,
                        height=512,
                        width=512,
                        guidance_scale=args.guidance,
                        latents=cand_lat,
                        generator=generator
                    ).images[0]

                cand_path = os.path.join(concept_cands_dir, f"p{pi}_c{ci}.png")
                img.save(cand_path)
                all_candidates[pi].append((img, cand_path))

        # E. CLIP MMR Selection
        print(f"  🎯 CLIP MMR 선별기 가동 중 (W_T={cfg['w_t']}, W_I={cfg['w_i']})...", flush=True)
        all_raw_refs = sorted(
            glob.glob(os.path.join(args.root, "dataset", concept, "*.png")) +
            glob.glob(os.path.join(args.root, "dataset", concept, "*.jpg"))
        )
        ref_rgb_list = [Image.open(p).convert("RGB") for p in all_raw_refs]

        te = F.normalize(clip_text_emb(clip_model, clip_proc, eval_prompts, device=device), dim=-1)
        ri = F.normalize(clip_image_emb(clip_model, clip_proc, ref_rgb_list, device=device), dim=-1)

        w_t = cfg.get("w_t", 1.0)
        w_i = cfg.get("w_i", 1.2)
        w_div = 0.35
        dup_th = 0.92
        dup_pen = 1.0

        chosen_embs = []
        selection_records = []

        for pi in range(len(raw_prompts)):
            cand_imgs = [item[0] for item in all_candidates[pi]]
            gi = F.normalize(clip_image_emb(clip_model, clip_proc, cand_imgs, device=device), dim=-1)

            s_t = gi @ te[pi]
            s_i = (gi @ ri.T).mean(dim=1)
            s_dup = (gi @ ri.T).max(dim=1).values

            score = w_t * s_t + w_i * s_i - dup_pen * (s_dup - dup_th).clamp(min=0)
            if chosen_embs:
                prev = torch.stack(chosen_embs)
                score = score - w_div * (gi @ prev.T).max(dim=1).values

            best_idx = int(score.argmax().item())
            chosen_embs.append(gi[best_idx])

            best_img, best_cand_p = all_candidates[pi][best_idx]
            final_p = os.path.join(concept_out_dir, f"{pi}.png")
            best_img.save(final_p)

            rec = {
                "prompt_idx": pi,
                "picked_candidate": best_idx,
                "clip_t": float(s_t[best_idx].item()),
                "clip_i": float(s_i[best_idx].item()),
                "dup": float(s_dup[best_idx].item()),
                "score": float(score[best_idx].item()),
            }
            selection_records.append(rec)
            print(f"    p{pi}: Picked c{best_idx} -> CLIP-T={rec['clip_t']:.4f}, CLIP-I={rec['clip_i']:.4f}, Score={rec['score']:.4f}", flush=True)

        with open(os.path.join(concept_out_dir, "selection.json"), "w", encoding="utf-8") as f:
            json.dump(selection_records, f, indent=2, ensure_ascii=False)

        # Unload LoRA adapter
        pipe.transformer = pipe.transformer.unload()
        torch.cuda.empty_cache()

    elapsed = round((time.time() - t0_all) / 60, 1)
    print(f"\n🎉 [Exp-11] 10개 서브젝트 100장 생성 및 선별 완료 (총 소요: {elapsed}분)!", flush=True)
    print(f"📁 최종 산출물 디렉토리: {args.output_dir}", flush=True)

    # 3. 공식 evaluation.py 실행
    print("\n📊 공식 evaluation.py 채점 수행 중...", flush=True)
    eval_script = os.path.join(args.root, "evaluation.py")
    cmd = f"python3 {eval_script} --dataset {os.path.join(args.root, 'dataset')} --prompts {os.path.join(args.root, 'prompt')} --images {args.output_dir}"
    os.system(cmd)


if __name__ == "__main__":
    main()
