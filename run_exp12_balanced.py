"""
Exp-12: Balanced SOTA Ensemble (Natural Reference + Background Permeability)
----------------------------------------------------------------------------
Fixes the white background washout from Exp-11:
1. Reference: Raw natural dataset images (dataset/{concept}/) with natural lighting.
2. Controlled ODE: Relaxed guidance (tau=0.58~0.65, eta=0.65~0.75) allowing full background synthesis.
3. Spherical Blend: Preserves unit Gaussian variance for razor-sharp textures.
4. Selection Objective: Balanced Pareto (W_T=1.5, W_I=0.9) actively rewarding scene compliance.
5. High-Speed 28-Step Euler Flow Matching Solver (~14 min total on A100).
"""

import argparse
import glob
import json
import math
import os
import shutil
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput
from peft import PeftModel

CLASS_PROMPT = {
    "actionfigure_2": "action figure",
    "decoritems_woodenpot": "wooden pot",
    "furniture_sofa2": "sofa",
    "instrument_music2": "guitar",
    "luggage_backpack1": "backpack",
    "person_3": "person",
    "pet_cat5": "cat",
    "scene_waterfall": "waterfall",
    "transport_tank": "tank",
    "wearable_jacket1": "jacket",
}

# Balanced guidance: keeps subject identity without locking background into solid white
SUBJECT_CONFIG = {
    "actionfigure_2":       {"tau": 0.58, "eta": 0.68, "w_t": 1.5, "w_i": 0.9},
    "decoritems_woodenpot": {"tau": 0.65, "eta": 0.75, "w_t": 1.5, "w_i": 0.9},
    "furniture_sofa2":      {"tau": 0.65, "eta": 0.75, "w_t": 1.5, "w_i": 0.9},
    "instrument_music2":    {"tau": 0.60, "eta": 0.70, "w_t": 1.5, "w_i": 0.9},
    "luggage_backpack1":    {"tau": 0.65, "eta": 0.72, "w_t": 1.5, "w_i": 0.9},
    "person_3":             {"tau": 0.58, "eta": 0.70, "w_t": 1.5, "w_i": 1.0},
    "pet_cat5":             {"tau": 0.60, "eta": 0.70, "w_t": 1.5, "w_i": 0.9},
    "scene_waterfall":      {"tau": 0.65, "eta": 0.75, "w_t": 1.5, "w_i": 0.9},
    "transport_tank":       {"tau": 0.58, "eta": 0.68, "w_t": 1.5, "w_i": 0.9},
    "wearable_jacket1":     {"tau": 0.60, "eta": 0.70, "w_t": 1.5, "w_i": 0.9},
}

DEFAULT_NEGATIVE_PROMPTS = {
    "default": "blurry, low quality, distorted, bad anatomy, flat background, white background, plain background",
    "person_3": "blurry, distorted face, bad anatomy, deformed eyes, double head, cloned face, unnatural skin, flat white background",
    "actionfigure_2": "blurry, deformed limbs, broken joints, melted plastic, distorted face, plain background, white background",
    "transport_tank": "blurry, distorted armor, melted tracks, extra barrels, deformed wheels, flat background, white background",
    "scene_waterfall": "blurry, low quality, ugly, dry river, distorted perspective, oversaturated, white background",
    "pet_cat5": "blurry, deformed paws, bad eyes, extra ears, mutated cat, cartoon, white background",
}


# ==============================================================================
# Schedulers
# ==============================================================================

class EulerControlledODE(FlowMatchEulerDiscreteScheduler):
    def set(self, reference: torch.Tensor, tau: float = 0.6, eta: float = 0.7, alpha: float = 1.5):
        self.reference = reference
        self.tau = float(tau)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self._step_index = None
        return self

    def controller(self, sample: torch.Tensor, sigma: torch.Tensor):
        reference = self.reference.to(device=sample.device, dtype=sample.dtype)
        return (sample - reference) / sigma.clamp_min(1e-6)

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, *args, return_dict: bool = True, **kwargs):
        if self.step_index is None:
            self._init_step_index(timestep)

        sample_fp32 = sample.to(torch.float32)
        sigma = self.sigmas[self.step_index].to(sample.device)
        sigma_next = self.sigmas[self.step_index + 1].to(sample.device)

        conditional_velocity = self.controller(sample_fp32, sigma).to(model_output.dtype)
        sigma_val = sigma.item()

        if sigma_val > self.tau:
            progress = (sigma_val - self.tau) / max(1.0 - self.tau, 1e-6)
            current_eta = self.eta * (progress ** self.alpha)
        else:
            current_eta = 0.0

        controlled_velocity = model_output + current_eta * (conditional_velocity - model_output)
        prev_sample = sample_fp32 + (sigma_next - sigma) * controlled_velocity
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)


class EulerControlledODEInversion(EulerControlledODE):
    def set(self, reference: torch.Tensor, tau: float = 0.0, eta: float = 0.5):
        return super().set(reference, tau=tau, eta=eta)

    def controller(self, sample: torch.Tensor, sigma: torch.Tensor):
        reference = self.reference.to(device=sample.device, dtype=sample.dtype)
        return (reference - sample) / (1.0 - sigma).clamp_min(1e-6)

    def set_timesteps(self, num_inference_steps: int = None, device=None, sigmas=None, mu=None, timesteps=None):
        super().set_timesteps(num_inference_steps=num_inference_steps, device=device, sigmas=sigmas, mu=mu, timesteps=timesteps)
        self.timesteps = torch.flip(self.timesteps, dims=(0,))
        self.sigmas = torch.flip(self.sigmas, dims=(0,))
        self._step_index = None


# ==============================================================================
# Helper Functions
# ==============================================================================

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


def fit_square(im: Image.Image, res: int = 512, mode: str = "pad") -> Image.Image:
    if mode == "crop":
        w, h = im.size
        k = res / min(w, h)
        im = im.resize((max(res, int(round(w * k))), max(res, int(round(h * k)))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - res) // 2, (h - res) // 2
        return im.crop((l, t, l + res, t + res))
    im = ImageOps.contain(im, (res, res), Image.BICUBIC)
    # Natural edge padding rather than pure white
    canvas = Image.new("RGB", (res, res), (128, 128, 128))
    canvas.paste(im, ((res - im.width) // 2, (res - im.height) // 2))
    return canvas


def get_natural_reference_latent(pipe: StableDiffusion3Pipeline, concept: str, root_dir: str, seed: int = 42, device: str = "cuda") -> Tuple[torch.Tensor, str]:
    raw_paths = sorted(
        glob.glob(os.path.join(root_dir, "dataset", concept, "*.png")) +
        glob.glob(os.path.join(root_dir, "dataset", concept, "*.jpg")) +
        glob.glob(os.path.join(root_dir, "dataset", concept, "*.jpeg"))
    )
    if not raw_paths:
        raise FileNotFoundError(f"No raw reference image found for {concept}")

    selected_path = raw_paths[0]
    print(f"  📸 자연 원본 참조 이미지 선택: {os.path.basename(selected_path)}", flush=True)
    img = Image.open(selected_path).convert("RGB")
    img = fit_square(img, 512, mode="pad")

    px = pipe.image_processor.preprocess(img).to(device=device, dtype=pipe.vae.dtype)
    post = pipe.vae.encode(px).latent_dist
    g = torch.Generator(device=device).manual_seed(seed)
    raw = post.sample(generator=g)
    shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
    scaling_factor = pipe.vae.config.scaling_factor
    return (raw - shift_factor) * scaling_factor, selected_path


def spherical_blend(anchor: torch.Tensor, seed: int, strength: float = 0.20, device: str = "cuda") -> torch.Tensor:
    s = min(max(strength, 0.0), 0.95)
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(anchor.shape, generator=g, device=device, dtype=anchor.dtype)
    return math.sqrt(1.0 - s * s) * anchor + s * noise


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp-12: Balanced SOTA Ensemble")
    parser.add_argument("--root", type=str, default="/content/project-3")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/exp05_lora_hq")
    parser.add_argument("--output_dir", type=str, default="./experiments/12_balanced_ensemble")
    parser.add_argument("--candidates", type=int, default=4, help="후보 생성 수 (프롬프트당)")
    parser.add_argument("--steps", type=int, default=28, help="Euler 적분 스텝 수")
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("=" * 80, flush=True)
    print("🏆 [Exp-12] Balanced SOTA Ensemble (Natural Reference + Full Scene Context)", flush=True)
    print(f"• Checkpoints: {args.checkpoints_dir}", flush=True)
    print(f"• Output Directory: {args.output_dir}", flush=True)
    print(f"• Candidates per Prompt: N = {args.candidates} (총 400장 생성 후 최적 100장 선별)", flush=True)
    print(f"• Steps: {args.steps} Steps (High-Speed Euler Controlled ODE)", flush=True)
    print(f"• Reference: Raw natural dataset images (Fixes white background blowout)", flush=True)
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
        cfg = SUBJECT_CONFIG.get(concept, {"tau": 0.60, "eta": 0.70, "w_t": 1.5, "w_i": 0.9})

        # A. LoRA 주입
        lora_path = os.path.join(args.checkpoints_dir, f"lora_{concept}")
        if not os.path.exists(lora_path):
            lora_path = os.path.join(args.root, "checkpoints", "exp08_dreambooth_lora", f"lora_{concept}")
        print(f"  🔗 LoRA 가중치 주입: {lora_path}", flush=True)
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path, torch_dtype=dtype)

        # B. Natural Raw Reference Latent & Inversion
        ref_latent, ref_name = get_natural_reference_latent(
            pipe=pipe,
            concept=concept,
            root_dir=args.root,
            seed=args.seed,
            device=device
        )

        print(f"  📸 Natural Controlled ODE Inversion ({args.steps} Steps)...", flush=True)
        inversion_scheduler = EulerControlledODEInversion.from_config(base_sched_cfg)
        g_inv = torch.Generator(device=device).manual_seed(args.seed)
        prior_noise = torch.randn(ref_latent.shape, generator=g_inv, device=device, dtype=dtype)
        inversion_scheduler.set(reference=prior_noise, tau=0.0, eta=0.45)
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
        neg_prompt = DEFAULT_NEGATIVE_PROMPTS.get(concept, DEFAULT_NEGATIVE_PROMPTS["default"])

        concept_cands_dir = os.path.join(cands_dir, concept)
        os.makedirs(concept_cands_dir, exist_ok=True)
        concept_out_dir = os.path.join(args.output_dir, concept)
        os.makedirs(concept_out_dir, exist_ok=True)

        all_candidates = {pi: [] for pi in range(len(raw_prompts))}

        # D. 후보 생성 (N=4 candidates per prompt with Spherical Blend)
        print(f"  ⚡ 10개 프롬프트 × {args.candidates}개 후보 이미지 생성 중 (tau={cfg['tau']}, eta={cfg['eta']})...", flush=True)
        for pi, (gen_p, eval_p) in enumerate(zip(gen_prompts, eval_prompts)):
            for ci in range(args.candidates):
                cand_seed = args.seed + pi * 1000 + ci * 37
                strength = 0.12 + 0.14 * ci
                cand_lat = spherical_blend(inverted_latent, cand_seed, strength=strength, device=device)

                gen_scheduler = EulerControlledODE.from_config(base_sched_cfg)
                gen_scheduler.set(
                    reference=ref_latent,
                    tau=cfg["tau"],
                    eta=cfg["eta"],
                    alpha=1.5
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

        # E. CLIP MMR Selection (Balanced Pareto: W_T=1.5, W_I=0.9)
        w_t = cfg.get("w_t", 1.5)
        w_i = cfg.get("w_i", 0.9)
        w_div = 0.35
        dup_th = 0.92
        dup_pen = 1.0

        print(f"  🎯 CLIP MMR 선별기 가동 중 (W_T={w_t}, W_I={w_i})...", flush=True)
        all_raw_refs = sorted(
            glob.glob(os.path.join(args.root, "dataset", concept, "*.png")) +
            glob.glob(os.path.join(args.root, "dataset", concept, "*.jpg")) +
            glob.glob(os.path.join(args.root, "dataset", concept, "*.jpeg"))
        )
        ref_rgb_list = [Image.open(p).convert("RGB") for p in all_raw_refs]

        te = F.normalize(clip_text_emb(clip_model, clip_proc, eval_prompts, device=device), dim=-1)
        ri = F.normalize(clip_image_emb(clip_model, clip_proc, ref_rgb_list, device=device), dim=-1)

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

        # Save selection records outside concept image folder to prevent PIL Image.open conflict
        with open(os.path.join(args.output_dir, f"selection_{concept}.json"), "w", encoding="utf-8") as f:
            json.dump(selection_records, f, indent=2, ensure_ascii=False)

        # Unload LoRA adapter back to base model
        pipe.transformer = pipe.transformer.unload()
        torch.cuda.empty_cache()

    elapsed = round((time.time() - t0_all) / 60, 1)
    print(f"\n🎉 [Exp-12] 10개 서브젝트 100장 생성 및 선별 완료 (총 소요: {elapsed}분)!", flush=True)
    print(f"📁 최종 산출물 디렉토리: {args.output_dir}", flush=True)

    # 3. 공식 채점 실행
    print("\n" + "=" * 70, flush=True)
    print("📊 [Exp-12] 공식 evaluation.py 전체 10개 서브젝트 정밀 채점 시작", flush=True)
    print("=" * 70, flush=True)

    scores_summary = {"per_concept_scores": {}, "average_scores": {}}
    t2i_list, i2i_list = [], []

    eval_script = os.path.join(args.root, "evaluation.py")
    for concept, class_noun in CLASS_PROMPT.items():
        concept_dir = os.path.join(args.output_dir, concept)
        if not os.path.exists(concept_dir):
            continue
        cmd = f"{sys.executable} {eval_script} --dataset {os.path.join(args.root, 'dataset')} --prompts {os.path.join(args.root, 'prompt')} --concept {concept} --images {concept_dir}"
        print(f"\n--- [{concept}] 채점 ---", flush=True)
        os.system(cmd)

        prompts_file = os.path.join(args.root, "prompt", f"{concept}.txt")
        with open(prompts_file, "r", encoding="utf-8") as f:
            c_prompts = [l.strip().replace("{}", class_noun) for l in f.readlines() if l.strip()]

        gen_imgs = [Image.open(os.path.join(concept_dir, f"{i}.png")).convert("RGB") for i in range(len(c_prompts))]
        ref_imgs = [Image.open(p).convert("RGB") for p in sorted(glob.glob(os.path.join(args.root, "dataset", concept, "*.*")))]

        te = F.normalize(clip_text_emb(clip_model, clip_proc, c_prompts, device=device), dim=-1)
        gi = F.normalize(clip_image_emb(clip_model, clip_proc, gen_imgs, device=device), dim=-1)
        ri = F.normalize(clip_image_emb(clip_model, clip_proc, ref_imgs, device=device), dim=-1)

        c_t2i = float(F.cosine_similarity(gi, te).mean().item())
        c_i2i = float((gi @ ri.T).mean().item())
        scores_summary["per_concept_scores"][concept] = {"t2i": c_t2i, "i2i": c_i2i}
        t2i_list.append(c_t2i)
        i2i_list.append(c_i2i)

    mean_t2i = float(np.mean(t2i_list))
    mean_i2i = float(np.mean(i2i_list))
    scores_summary["average_scores"] = {"t2i": mean_t2i, "i2i": mean_i2i, "total": mean_t2i + mean_i2i}

    with open(os.path.join(args.output_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(scores_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70, flush=True)
    print(f"🏆 [Exp-12 최종 공식 결과] CLIP-T: {mean_t2i:.4f} | CLIP-I: {mean_i2i:.4f} | TOTAL: {mean_t2i + mean_i2i:.4f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
