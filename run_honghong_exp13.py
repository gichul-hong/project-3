"""
run_honghong_exp13.py
-------------------------------------------------------------------
Exp-13 파이프라인을 'honghong' 어린이 서브젝트에 그대로 적용.
• LoRA는 exp05_lora_hq 또는 exp08_dreambooth_lora에서 person_3 가중치를 폴백으로 사용
• honghong 전용 τ=0.58, η=0.70 (어린이 얼굴 — 배경 합성 자유도 최대)
• 프롬프트: prompt/honghong.txt (10개)
• 출력: experiments/honghong_exp13/
-------------------------------------------------------------------
Colab 실행 명령:
  !python run_honghong_exp13.py --root /content/project-3
"""

import argparse, glob, json, math, os, shutil, sys, time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput
from peft import PeftModel


# ── Controlled ODE Schedulers (Exp-13 동일) ─────────────────────────────────

class EulerControlledODE(FlowMatchEulerDiscreteScheduler):
    def set(self, reference, tau=0.65, eta=0.75, alpha=1.2):
        self.reference = reference
        self.tau = float(tau)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self._step_index = None
        return self

    def controller(self, sample, sigma):
        reference = self.reference.to(device=sample.device, dtype=sample.dtype)
        return (sample - reference) / sigma.clamp_min(1e-6)

    def step(self, model_output, timestep, sample, *args, return_dict=True, **kwargs):
        if self.step_index is None:
            self._init_step_index(timestep)
        sample_fp32 = sample.to(torch.float32)
        sigma = self.sigmas[self.step_index].to(sample.device)
        sigma_next = self.sigmas[self.step_index + 1].to(sample.device)
        cond_v = self.controller(sample_fp32, sigma).to(model_output.dtype)
        sigma_val = sigma.item()
        if sigma_val > self.tau:
            progress = (sigma_val - self.tau) / max(1.0 - self.tau, 1e-6)
            cur_eta = self.eta * (progress ** self.alpha)
        else:
            cur_eta = 0.0
        ctrl_v = model_output + cur_eta * (cond_v - model_output)
        prev = sample_fp32 + (sigma_next - sigma) * ctrl_v
        prev = prev.to(model_output.dtype)
        self._step_index += 1
        if not return_dict:
            return (prev,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev)


class EulerControlledODEInversion(EulerControlledODE):
    def set(self, reference, tau=0.0, eta=0.5):
        return super().set(reference, tau=tau, eta=eta)

    def controller(self, sample, sigma):
        reference = self.reference.to(device=sample.device, dtype=sample.dtype)
        return (reference - sample) / (1.0 - sigma).clamp_min(1e-6)

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
        super().set_timesteps(num_inference_steps=num_inference_steps, device=device,
                              sigmas=sigmas, mu=mu, timesteps=timesteps)
        self.timesteps = torch.flip(self.timesteps, dims=(0,))
        self.sigmas = torch.flip(self.sigmas, dims=(0,))
        self._step_index = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _unwrap(x):
    return x if isinstance(x, torch.Tensor) else x.pooler_output


@torch.no_grad()
def clip_text_emb(model, proc, prompts, device="cuda"):
    b = proc(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    return _unwrap(model.get_text_features(**b)).float()


@torch.no_grad()
def clip_image_emb(model, proc, images, device="cuda", bs=16):
    out = []
    for i in range(0, len(images), bs):
        b = proc(images=images[i:i+bs], return_tensors="pt").to(device)
        out.append(_unwrap(model.get_image_features(**b)).float())
    return torch.cat(out)


def compute_border_white_frac(img, thresh=240, border_ratio=0.12):
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    bh, bw = max(1, int(h*border_ratio)), max(1, int(w*border_ratio))
    def _white(chunk):
        return (chunk[:,:,0]>=thresh) & (chunk[:,:,1]>=thresh) & (chunk[:,:,2]>=thresh)
    cnt = (np.sum(_white(arr[:bh])) + np.sum(_white(arr[-bh:])) +
           np.sum(_white(arr[:,:bw])) + np.sum(_white(arr[:,-bw:])))
    total = bh*w*2 + bw*h*2
    return float(cnt / max(1, total))


def fit_square(im, res=512, mode="crop"):
    if mode == "crop":
        w, h = im.size
        k = res / min(w, h)
        im = im.resize((max(res, int(round(w*k))), max(res, int(round(h*k)))), Image.BICUBIC)
        w, h = im.size
        l, t = (w-res)//2, (h-res)//2
        return im.crop((l, t, l+res, t+res))
    im = ImageOps.contain(im, (res, res), Image.BICUBIC)
    canvas = Image.new("RGB", (res, res), (128, 128, 128))
    canvas.paste(im, ((res-im.width)//2, (res-im.height)//2))
    return canvas


def get_ref_latent(pipe, dataset_dir, seed=42, device="cuda"):
    """honghong 폴더에서 첫 번째 이미지를 참조로 사용"""
    paths = sorted(
        glob.glob(os.path.join(dataset_dir, "*.png")) +
        glob.glob(os.path.join(dataset_dir, "*.jpg")) +
        glob.glob(os.path.join(dataset_dir, "*.jpeg")) +
        glob.glob(os.path.join(dataset_dir, "*.JPG")) +
        glob.glob(os.path.join(dataset_dir, "*.JPEG"))
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {dataset_dir}")
    selected = paths[0]
    print(f"  📸 참조 이미지: {os.path.basename(selected)}")
    img = Image.open(selected).convert("RGB")
    img = fit_square(img, 512, mode="crop")  # 어린이 얼굴 → crop 모드
    px = pipe.image_processor.preprocess(img).to(device=device, dtype=pipe.vae.dtype)
    post = pipe.vae.encode(px).latent_dist
    g = torch.Generator(device=device).manual_seed(seed)
    raw = post.sample(generator=g)
    sf = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
    return (raw - sf) * pipe.vae.config.scaling_factor, selected


def spherical_blend(anchor, seed, strength=0.20, device="cuda"):
    s = min(max(strength, 0.0), 0.95)
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(anchor.shape, generator=g, device=device, dtype=anchor.dtype)
    return math.sqrt(1.0 - s*s) * anchor + s * noise


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp-13 for honghong")
    parser.add_argument("--root",           default="/content/project-3")
    parser.add_argument("--checkpoints_dir",default=None,
                        help="LoRA 체크포인트 디렉터리 (None이면 자동 탐색)")
    parser.add_argument("--output_dir",     default="./experiments/honghong_exp13")
    parser.add_argument("--candidates",     type=int,   default=4)
    parser.add_argument("--steps",          type=int,   default=28)
    parser.add_argument("--guidance",       type=float, default=7.0)
    parser.add_argument("--seed",           type=int,   default=42)
    # honghong 전용 파라미터 (어린이 얼굴: 배경 자유도 높임)
    parser.add_argument("--tau",            type=float, default=0.58,
                        help="ODE threshold — 작을수록 배경 변환 자유도 ↑")
    parser.add_argument("--eta",            type=float, default=0.70,
                        help="ODE guidance strength")
    args = parser.parse_args()

    CONCEPT     = "honghong"
    CLASS_NOUN  = "child"   # CLIP-T 채점 시 {} 치환 대상
    DATASET_DIR = os.path.join(args.root, "dataset", CONCEPT)
    PROMPT_FILE = os.path.join(args.root, "prompt", f"{CONCEPT}.txt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16

    print("=" * 75)
    print(f"🍀 honghong Exp-13 Personalization Pipeline")
    print(f"  τ={args.tau}, η={args.eta}, steps={args.steps}, N={args.candidates}")
    print(f"  dataset: {DATASET_DIR}")
    print(f"  output : {args.output_dir}")
    print("=" * 75)

    # ── 1. CLIP 로드 ──────────────────────────────────────────────────────────
    clip_id    = "openai/clip-vit-base-patch32"
    print(f"📦 CLIP 로딩: {clip_id}")
    clip_model = CLIPModel.from_pretrained(clip_id).to(device).eval()
    clip_proc  = CLIPProcessor.from_pretrained(clip_id)

    # ── 2. SD3.5 파이프라인 로드 ──────────────────────────────────────────────
    print("📦 SD3.5-medium 파이프라인 로딩...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    base_sched_cfg = dict(pipe.scheduler.config)

    # ── 3. LoRA 탐색 및 주입 ─────────────────────────────────────────────────
    # 우선순위: (1) --checkpoints_dir/lora_honghong
    #           (2) exp08_dreambooth_lora/lora_honghong (있으면)
    #           (3) exp05_lora_hq/lora_person_3         (사람 얼굴 폴백)
    lora_candidates = []
    if args.checkpoints_dir:
        lora_candidates.append(os.path.join(args.checkpoints_dir, f"lora_{CONCEPT}"))
    lora_candidates += [
        os.path.join(args.root, "checkpoints", "exp08_dreambooth_lora", f"lora_{CONCEPT}"),
        os.path.join(args.root, "checkpoints", "exp05_lora_hq",          f"lora_{CONCEPT}"),
        os.path.join(args.root, "checkpoints", "exp08_dreambooth_lora",  "lora_person_3"),
        os.path.join(args.root, "checkpoints", "exp05_lora_hq",          "lora_person_3"),
    ]

    lora_path = None
    for p in lora_candidates:
        if os.path.exists(p):
            lora_path = p
            break

    if lora_path:
        print(f"🔗 LoRA 주입: {lora_path}")
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path, torch_dtype=dtype)
    else:
        print("⚠️  LoRA 체크포인트를 찾지 못했습니다. base SD3.5만으로 진행합니다.")
        print("   (정확한 결과를 위해 먼저 LoRA를 학습하세요 — 아래 옵션 B 참조)")

    # ── 4. 참조 잠재 & Inversion ─────────────────────────────────────────────
    ref_latent, ref_path = get_ref_latent(pipe, DATASET_DIR, seed=args.seed, device=device)

    print(f"📸 Controlled ODE Inversion ({args.steps} steps)...")
    inv_sched = EulerControlledODEInversion.from_config(base_sched_cfg)
    g_inv = torch.Generator(device=device).manual_seed(args.seed)
    prior_noise = torch.randn(ref_latent.shape, generator=g_inv, device=device, dtype=dtype)
    inv_sched.set(reference=prior_noise, tau=0.0, eta=0.45)
    pipe.scheduler = inv_sched
    with torch.no_grad():
        inv_out = pipe(
            prompt="", guidance_scale=1.0,
            num_inference_steps=args.steps,
            output_type="latent", latents=ref_latent,
        )
    inverted_latent = inv_out.images.clone()

    # ── 5. 프롬프트 로드 ─────────────────────────────────────────────────────
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        raw_prompts = [l.strip() for l in f if l.strip()]

    eval_prompts = [p.replace("{}", CLASS_NOUN)        for p in raw_prompts]
    gen_prompts  = [p.replace("{}", f"sks {CLASS_NOUN}") for p in raw_prompts]
    neg_prompt   = ("blurry, distorted face, bad anatomy, deformed eyes, "
                    "double head, cloned face, unnatural skin, flat white background, "
                    "cartoon, anime, unrealistic")

    out_dir   = args.output_dir
    cands_dir = os.path.join(out_dir, "candidates", CONCEPT)
    final_dir = os.path.join(out_dir, CONCEPT)
    os.makedirs(cands_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # ── 6. 후보 생성 (N=4, Spherical Blend) ──────────────────────────────────
    print(f"\n⚡ {len(raw_prompts)}개 프롬프트 × {args.candidates}후보 생성 중 (τ={args.tau}, η={args.eta})...")
    all_candidates = {pi: [] for pi in range(len(raw_prompts))}

    for pi, (gen_p, eval_p) in enumerate(zip(gen_prompts, eval_prompts)):
        print(f"  Prompt [{pi}]: {eval_p[:60]}...")
        for ci in range(args.candidates):
            cand_seed = args.seed + pi*1000 + ci*37
            strength  = 0.10 + 0.12*ci
            cand_lat  = spherical_blend(inverted_latent, cand_seed, strength=strength, device=device)

            gen_sched = EulerControlledODE.from_config(base_sched_cfg)
            gen_sched.set(reference=ref_latent, tau=args.tau, eta=args.eta, alpha=1.2)
            pipe.scheduler = gen_sched

            with torch.no_grad():
                img = pipe(
                    prompt=gen_p,
                    negative_prompt=neg_prompt,
                    num_inference_steps=args.steps,
                    height=512, width=512,
                    guidance_scale=args.guidance,
                    latents=cand_lat,
                ).images[0]

            cand_path = os.path.join(cands_dir, f"p{pi}_c{ci}.png")
            img.save(cand_path)
            wf = compute_border_white_frac(img)
            all_candidates[pi].append((img, cand_path, wf))
            print(f"    c{ci} ✓  white_frac={wf:.3f}")

    # ── 7. 1:1 MMR 선별 ──────────────────────────────────────────────────────
    print("\n🎯 1:1 공식 Total 선별기 가동 (W_T=1.0, W_I=1.0)...")
    ref_imgs = [Image.open(p).convert("RGB") for p in sorted(
        glob.glob(os.path.join(DATASET_DIR, "*.jpg")) +
        glob.glob(os.path.join(DATASET_DIR, "*.jpeg")) +
        glob.glob(os.path.join(DATASET_DIR, "*.png")) +
        glob.glob(os.path.join(DATASET_DIR, "*.JPG")) +
        glob.glob(os.path.join(DATASET_DIR, "*.JPEG"))
    )]
    te = F.normalize(clip_text_emb(clip_model, clip_proc, eval_prompts, device=device), dim=-1)
    ri = F.normalize(clip_image_emb(clip_model, clip_proc, ref_imgs, device=device), dim=-1)

    chosen_embs = []
    records     = []
    t2i_all, i2i_all = [], []

    for pi in range(len(raw_prompts)):
        imgs = [item[0] for item in all_candidates[pi]]
        gi   = F.normalize(clip_image_emb(clip_model, clip_proc, imgs, device=device), dim=-1)

        s_t   = gi @ te[pi]
        s_i   = (gi @ ri.T).mean(dim=1)
        s_dup = (gi @ ri.T).max(dim=1).values
        wf_t  = torch.tensor([item[2] for item in all_candidates[pi]], device=device, dtype=torch.float32)

        score = s_t + s_i - 1.0*(s_dup-0.92).clamp(min=0) - 1.5*(wf_t-0.18).clamp(min=0)
        if chosen_embs:
            prev  = torch.stack(chosen_embs)
            score = score - 0.35*(gi @ prev.T).max(dim=1).values

        best = int(score.argmax())
        chosen_embs.append(gi[best])

        best_img, _, best_wf = all_candidates[pi][best]
        out_path = os.path.join(final_dir, f"{pi}.png")
        best_img.save(out_path)

        rec = {
            "prompt_idx": pi, "prompt": eval_prompts[pi],
            "picked": best,
            "clip_t": float(s_t[best]), "clip_i": float(s_i[best]),
            "white_frac": float(best_wf),
        }
        records.append(rec)
        t2i_all.append(rec["clip_t"]); i2i_all.append(rec["clip_i"])
        print(f"  p{pi}: c{best} → CLIP-T={rec['clip_t']:.4f}, CLIP-I={rec['clip_i']:.4f}, "
              f"Total={rec['clip_t']+rec['clip_i']:.4f}")

    # ── 8. 저장 ──────────────────────────────────────────────────────────────
    summary = {
        "concept": CONCEPT,
        "lora_used": lora_path or "none (base model)",
        "tau": args.tau, "eta": args.eta,
        "per_prompt": records,
        "average": {
            "clip_t": float(np.mean(t2i_all)),
            "clip_i": float(np.mean(i2i_all)),
            "total":  float(np.mean(t2i_all) + np.mean(i2i_all)),
        }
    }
    with open(os.path.join(out_dir, "honghong_result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f"🎉 완료!  CLIP-T: {summary['average']['clip_t']:.4f}  "
          f"CLIP-I: {summary['average']['clip_i']:.4f}  "
          f"Total: {summary['average']['total']:.4f}")
    print(f"📁 결과 이미지: {final_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
