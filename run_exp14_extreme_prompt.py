"""
================================================================================
🚀 [Exp-14] Extreme Prompt Alignment & CLIP-T Maximization Pipeline
================================================================================
목표:
  - 프롬프트 텍스트 충실도(CLIP-T)를 극대화하는 극한의 텍스트 정렬 생성.
  - 레퍼런스 이미지의 과도한 고착을 완화(tau=0.42~0.48, eta=0.45~0.52)하고,
    CFG Guidance Scale을 7.5로 대폭 상향하여 프롬프트의 배경/광원/행동 렌더링을 100% 지배.
  - 선별 목적함수: Score = 2.0 * CLIP-T + 0.1 * CLIP-I - 2.0 * WhitePen
    (텍스트 일치도에 20배 높은 가중치 부여)
================================================================================
"""

import os
import sys
import glob
import json
import time
import math
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput
from peft import PeftModel
from transformers import CLIPProcessor, CLIPModel

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

# Exp-14 전용 극한의 텍스트 개방형 파라미터 (낮은 tau, eta)
SUBJECT_CONFIG = {
    "actionfigure_2":      {"tau": 0.42, "eta": 0.48, "fit_mode": "crop"},
    "decoritems_woodenpot":{"tau": 0.46, "eta": 0.50, "fit_mode": "crop"},
    "furniture_sofa2":     {"tau": 0.46, "eta": 0.50, "fit_mode": "crop"},
    "instrument_music2":   {"tau": 0.45, "eta": 0.50, "fit_mode": "crop"},
    "luggage_backpack1":   {"tau": 0.44, "eta": 0.48, "fit_mode": "crop"},
    "person_3":            {"tau": 0.40, "eta": 0.45, "fit_mode": "crop"},
    "pet_cat5":            {"tau": 0.44, "eta": 0.48, "fit_mode": "crop"},
    "scene_waterfall":     {"tau": 0.45, "eta": 0.50, "fit_mode": "pad"},
    "transport_tank":      {"tau": 0.40, "eta": 0.45, "fit_mode": "crop"},
    "wearable_jacket1":    {"tau": 0.44, "eta": 0.48, "fit_mode": "crop"},
}

# ==============================================================================
# Euler Controlled ODE Schedulers
# ==============================================================================
class EulerControlledODE(FlowMatchEulerDiscreteScheduler):
    def set(self, reference: torch.Tensor, tau: float = 0.45, eta: float = 0.50, alpha: float = 1.2):
        self.reference = reference
        self.tau = tau
        self.eta = eta
        self.alpha = alpha

    def controller(self, sample: torch.Tensor, sigma: torch.Tensor):
        reference = self.reference.to(device=sample.device, dtype=sample.dtype)
        return (sample - reference) / sigma.clamp_min(1e-6)

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, return_dict: bool = True):
        if self._step_index is None:
            self._init_step_index(timestep)

        sample_fp32 = sample.to(torch.float32)
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        sigma_val = sigma.item()

        conditional_velocity = self.controller(sample_fp32, sigma)

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
    def set(self, reference: torch.Tensor, tau: float = 0.0, eta: float = 0.40):
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

def compute_border_white_frac(img: Image.Image, thresh: int = 240, border_ratio: float = 0.12) -> float:
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    bh, bw = max(1, int(h * border_ratio)), max(1, int(w * border_ratio))
    top = arr[:bh, :, :]
    bottom = arr[-bh:, :, :]
    left = arr[:, :bw, :]
    right = arr[:, -bw:, :]
    def _is_white(chunk):
        return (chunk[:, :, 0] >= thresh) & (chunk[:, :, 1] >= thresh) & (chunk[:, :, 2] >= thresh)
    white_cnt = np.sum(_is_white(top)) + np.sum(_is_white(bottom)) + np.sum(_is_white(left)) + np.sum(_is_white(right))
    total_cnt = top.shape[0]*top.shape[1] + bottom.shape[0]*bottom.shape[1] + left.shape[0]*left.shape[1] + right.shape[0]*right.shape[1]
    return float(white_cnt / max(1, total_cnt))

def fit_square_custom(im: Image.Image, res: int = 512, mode: str = "crop") -> Image.Image:
    if mode == "crop":
        w, h = im.size
        k = res / min(w, h)
        im = im.resize((max(res, int(round(w * k))), max(res, int(round(h * k)))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - res) // 2, (h - res) // 2
        return im.crop((l, t, l + res, t + res))
    im = ImageOps.contain(im, (res, res), Image.BICUBIC)
    canvas = Image.new("RGB", (res, res), (128, 128, 128))
    canvas.paste(im, ((res - im.width) // 2, (res - im.height) // 2))
    return canvas

def get_reference_latent_exp14(pipe: StableDiffusion3Pipeline, concept: str, root_dir: str, fit_mode: str = "crop", seed: int = 42, device: str = "cuda") -> Tuple[torch.Tensor, str]:
    raw_paths = sorted(
        glob.glob(os.path.join(root_dir, "dataset", concept, "*.png")) +
        glob.glob(os.path.join(root_dir, "dataset", concept, "*.jpg")) +
        glob.glob(os.path.join(root_dir, "dataset", concept, "*.jpeg"))
    )
    if not raw_paths:
        raise FileNotFoundError(f"No raw reference image found for {concept}")

    selected_path = raw_paths[0]
    print(f"  📸 참조 이미지 선택 ({fit_mode} 모드): {os.path.basename(selected_path)}", flush=True)
    img = Image.open(selected_path).convert("RGB")
    img = fit_square_custom(img, 512, mode=fit_mode)

    px = pipe.image_processor.preprocess(img).to(device=device, dtype=pipe.vae.dtype)
    post = pipe.vae.encode(px).latent_dist
    g = torch.Generator(device=device).manual_seed(seed)
    raw = post.sample(generator=g)
    shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
    scaling_factor = pipe.vae.config.scaling_factor
    return (raw - shift_factor) * scaling_factor, selected_path

def spherical_blend(anchor: torch.Tensor, seed: int, strength: float = 0.28, device: str = "cuda") -> torch.Tensor:
    s = min(max(strength, 0.0), 0.95)
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(anchor.shape, generator=g, device=device, dtype=anchor.dtype)
    return math.sqrt(1.0 - s * s) * anchor + s * noise

# ==============================================================================
# Main Pipeline
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Exp-14: Extreme Prompt Alignment")
    parser.add_argument("--root", type=str, default="/content/project-3")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/exp05_lora_hq")
    parser.add_argument("--output_dir", type=str, default="./experiments/14_extreme_prompt_align")
    parser.add_argument("--candidates", type=int, default=4, help="후보 생성 수 (프롬프트당)")
    parser.add_argument("--steps", type=int, default=28, help="Euler 적분 스텝 수")
    parser.add_argument("--guidance", type=float, default=7.5, help="극한의 프롬프트 가이던스")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_done", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("=" * 85, flush=True)
    print("🚀 [Exp-14] Extreme Prompt Alignment & CLIP-T Maximization Pipeline", flush=True)
    print(f"• Checkpoints: {args.checkpoints_dir}", flush=True)
    print(f"• Output Directory: {args.output_dir}", flush=True)
    print(f"• CFG Guidance Scale: {args.guidance} (High Prompt Fidelity)", flush=True)
    print(f"• Selection Objective: Pure CLIP-T Maximizer (W_T=2.0, W_I=0.1, WhitePen=2.0)", flush=True)
    print("=" * 85, flush=True)

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
        concept_out_dir = os.path.join(args.output_dir, concept)
        if args.skip_done and os.path.exists(concept_out_dir) and len(glob.glob(os.path.join(concept_out_dir, "*.png"))) == 10:
            print(f"⏭️ [{concept}] 이미 10장 완료됨 -> 스킵", flush=True)
            continue

        print(f"\n{'='*75}\n▶ [{concept}] Processing (Class: '{class_noun}')...\n{'='*75}", flush=True)
        cfg = SUBJECT_CONFIG.get(concept, {"tau": 0.44, "eta": 0.48, "fit_mode": "crop"})

        # A. LoRA 주입
        lora_path = os.path.join(args.checkpoints_dir, f"lora_{concept}")
        if not os.path.exists(lora_path):
            lora_path = os.path.join(args.root, "checkpoints", "exp05_lora_hq", f"lora_{concept}")
        print(f"  🔗 LoRA 가중치 주입: {lora_path}", flush=True)
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path, torch_dtype=dtype)

        # B. Reference Latent & Inversion
        ref_latent, ref_name = get_reference_latent_exp14(
            pipe=pipe,
            concept=concept,
            root_dir=args.root,
            fit_mode=cfg.get("fit_mode", "crop"),
            seed=args.seed,
            device=device
        )

        print(f"  📸 Controlled ODE Inversion ({args.steps} Steps)...", flush=True)
        inversion_scheduler = EulerControlledODEInversion.from_config(base_sched_cfg)
        g_inv = torch.Generator(device=device).manual_seed(args.seed)
        prior_noise = torch.randn(ref_latent.shape, generator=g_inv, device=device, dtype=dtype)
        inversion_scheduler.set(reference=prior_noise, tau=0.0, eta=0.40)
        pipe.scheduler = inversion_scheduler

        with torch.no_grad():
            inv_out = pipe(
                prompt="",
                guidance_scale=1.0,
                num_inference_steps=args.steps,
                latents=ref_latent,
                output_type="latent",
            )
        anchor_latent = inv_out.images.detach()

        # C. Load Prompts
        prompt_file = os.path.join(args.root, "prompt", f"{concept}.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [l.strip().replace("{}", class_noun) for l in f.readlines() if l.strip()]

        raw_ref_imgs = [Image.open(p).convert("RGB") for p in sorted(glob.glob(os.path.join(args.root, "dataset", concept, "*.*")))]
        clean_ref_imgs = [fit_square_custom(im, 512, mode="crop") for im in raw_ref_imgs]

        ref_feat = clip_image_emb(clip_model, clip_proc, clean_ref_imgs, device=device)
        ref_feat = F.normalize(ref_feat, dim=-1)

        # D. Generate Candidates & Extreme CLIP-T Reranking
        tau, eta = cfg["tau"], cfg["eta"]
        print(f"  ⚡ 10개 프롬프트 × {args.candidates}개 후보 이미지 생성 중 (tau={tau}, eta={eta}, CFG={args.guidance})...", flush=True)

        cand_subj_dir = os.path.join(cands_dir, concept)
        os.makedirs(cand_subj_dir, exist_ok=True)
        os.makedirs(concept_out_dir, exist_ok=True)

        forward_scheduler = EulerControlledODE.from_config(base_sched_cfg)
        forward_scheduler.set(reference=ref_latent, tau=tau, eta=eta, alpha=1.2)
        pipe.scheduler = forward_scheduler

        picked_records = {}
        t_concept_start = time.time()

        for p_idx, prompt in enumerate(prompts):
            txt_feat = clip_text_emb(clip_model, clip_proc, [prompt], device=device)
            txt_feat = F.normalize(txt_feat, dim=-1)

            cands_pil = []
            for c_idx in range(args.candidates):
                cand_seed = args.seed + p_idx * 100 + c_idx * 13
                init_lat = spherical_blend(anchor_latent, seed=cand_seed, strength=0.28, device=device)

                forward_scheduler.set(reference=ref_latent, tau=tau, eta=eta, alpha=1.2)
                g_gen = torch.Generator(device=device).manual_seed(cand_seed)

                with torch.no_grad():
                    gen_out = pipe(
                        prompt=prompt,
                        guidance_scale=args.guidance,
                        num_inference_steps=args.steps,
                        latents=init_lat,
                        generator=g_gen,
                    )
                img = gen_out.images[0]
                cand_path = os.path.join(cand_subj_dir, f"p{p_idx}_c{c_idx}.png")
                img.save(cand_path)
                cands_pil.append(img)

            # CLIP Scores for Candidates
            c_feats = clip_image_emb(clip_model, clip_proc, cands_pil, device=device)
            c_feats = F.normalize(c_feats, dim=-1)

            t2i_scores = (c_feats @ txt_feat.T).squeeze(-1).cpu().tolist()
            if isinstance(t2i_scores, float): t2i_scores = [t2i_scores]
            
            i2i_matrix = (c_feats @ ref_feat.T).cpu()
            i2i_scores = i2i_matrix.mean(dim=-1).tolist()
            if isinstance(i2i_scores, float): i2i_scores = [i2i_scores]

            cand_records = []
            for c_idx, (t_sc, i_sc, c_img) in enumerate(zip(t2i_scores, i2i_scores, cands_pil)):
                bw_frac = compute_border_white_frac(c_img, thresh=240, border_ratio=0.12)
                white_pen = 2.0 * max(0.0, bw_frac - 0.18)
                
                # Extreme CLIP-T Maximizer Formula:
                obj_score = 2.0 * t_sc + 0.1 * i_sc - white_pen
                
                cand_records.append({
                    "cand_idx": c_idx,
                    "clip_t": float(t_sc),
                    "clip_i": float(i_sc),
                    "white_frac": float(bw_frac),
                    "white_pen": float(white_pen),
                    "obj_score": float(obj_score)
                })

            cand_records.sort(key=lambda x: x["obj_score"], reverse=True)
            best_cand = cand_records[0]
            best_idx = best_cand["cand_idx"]

            # Save Top-1 Selected Image
            final_path = os.path.join(concept_out_dir, f"{p_idx}.png")
            cands_pil[best_idx].save(final_path)

            picked_records[f"p{p_idx}"] = {
                "picked_candidate": best_idx,
                "clip_t": best_cand["clip_t"],
                "clip_i": best_cand["clip_i"],
                "obj_score": best_cand["obj_score"],
                "all_candidates": cand_records
            }
            print(f"    p{p_idx}: Picked c{best_idx} -> CLIP-T={best_cand['clip_t']:.4f}, CLIP-I={best_cand['clip_i']:.4f} (ObjScore={best_cand['obj_score']:.4f})", flush=True)

        with open(os.path.join(args.output_dir, f"selection_{concept}.json"), "w", encoding="utf-8") as f:
            json.dump(picked_records, f, indent=2, ensure_ascii=False)

        # LoRA 언로드
        pipe.transformer = pipe.transformer.unload()

    t_total = (time.time() - t0_all) / 60
    print(f"\n🎉 [Exp-14] 10개 서브젝트 100장 생성 및 선별 완료 (총 소요: {t_total:.1f}분)!", flush=True)

    # 6. Official Evaluation
    print(f"\n{'='*75}\n📊 [Exp-14] 공식 evaluation.py 전체 10개 서브젝트 정밀 채점 시작\n{'='*75}\n", flush=True)
    os.system(f"python evaluate.py --exp_dir {args.output_dir}")

if __name__ == "__main__":
    main()
