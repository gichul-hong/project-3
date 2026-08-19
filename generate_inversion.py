"""
Subject-driven Inversion & Customization Pipeline for SD3.5-Medium
------------------------------------------------------------------
DEVELOPMENT_GUIDE.md 및 Day1 실습 기반:
1. 레퍼런스 이미지 VAE Latent 인코딩
2. Inversion (RF-Inversion / Euler Inversion)
3. Controlled ODE Generation (10개 테스트 프롬프트 -> 0.png ~ 9.png 생성)
4. CLIP-T / CLIP-I 성능 평가 및 JSON/Markdown 결과 자동 저장
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from diffusers import (
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput,
)

# .env 파일에서 HF_TOKEN 로드
load_dotenv()

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


# ==============================================================================
# 1. Inversion & Generation Custom Schedulers (Flow Matching / Controlled ODE)
# ==============================================================================

class EulerInversion(FlowMatchEulerDiscreteScheduler):
    """FlowMatch Euler 스케줄을 data -> noise 방향으로 시간 역전시키는 Scheduler"""

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device = None,
        sigmas = None,
        mu = None,
        timesteps = None,
    ):
        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )
        self.timesteps = torch.flip(self.timesteps, dims=(0,))
        self.sigmas = torch.flip(self.sigmas, dims=(0,))


class ControlledODE(FlowMatchEulerDiscreteScheduler):
    """원본 레퍼런스 latent 방향의 conditional velocity를 보정/혼합하는 생성 Scheduler (RF-Inversion)"""

    def set(self, reference: torch.Tensor, tau: float = 0.7, eta: float = 0.9):
        self.reference = reference
        self.tau = float(tau)
        self.eta = float(eta)
        self._step_index = None
        return self

    def controller(self, sample: torch.Tensor, sigma: torch.Tensor):
        reference = self.reference.to(device=sample.device, dtype=sample.dtype)
        return (sample - reference) / sigma.clamp_min(1e-6)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *args,
        return_dict: bool = True,
        **kwargs,
    ):
        if self.step_index is None:
            self._init_step_index(timestep)

        sample_fp32 = sample.to(torch.float32)
        sigma = self.sigmas[self.step_index].to(sample.device)
        sigma_next = self.sigmas[self.step_index + 1].to(sample.device)

        conditional_velocity = self.controller(sample_fp32, sigma).to(model_output.dtype)
        current_eta = self.eta if sigma.item() > self.tau else 0.0

        controlled_velocity = model_output + current_eta * (conditional_velocity - model_output)
        prev_sample = sample_fp32 + (sigma_next - sigma) * controlled_velocity
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)


class ControlledODEInversion(ControlledODE):
    """Sampled prior (Noise) 방향의 conditional velocity를 보정하는 역변환 Scheduler"""

    def set(self, reference: torch.Tensor, tau: float = 0.0, eta: float = 0.5):
        return super().set(reference, tau=tau, eta=eta)

    def controller(self, sample: torch.Tensor, sigma: torch.Tensor):
        reference = self.reference.to(device=sample.device, dtype=sample.dtype)
        return (reference - sample) / (1.0 - sigma).clamp_min(1e-6)

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device = None,
        sigmas = None,
        mu = None,
        timesteps = None,
    ):
        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )
        self.timesteps = torch.flip(self.timesteps, dims=(0,))
        self.sigmas = torch.flip(self.sigmas, dims=(0,))
        self._step_index = None


# ==============================================================================
# 2. VAE Encoding Helper
# ==============================================================================

@torch.no_grad()
def encode_image_to_sd3_latent(pipe: StableDiffusion3Pipeline, image: Image.Image, seed: int = 0) -> torch.Tensor:
    """PIL 이미지를 전처리하여 SD3 VAE Latent로 변환 및 스케일링 적용"""
    device = pipe._execution_device
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((512, 512), Image.Resampling.BICUBIC)

    image_tensor = pipe.image_processor.preprocess(image).to(
        device=device, dtype=pipe.vae.dtype
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    posterior = pipe.vae.encode(image_tensor).latent_dist
    raw_latent = posterior.sample(generator=generator)

    shift_factor = pipe.vae.config.shift_factor
    scaling_factor = pipe.vae.config.scaling_factor
    return (raw_latent - shift_factor) * scaling_factor


def get_reference_latents(
    pipe: StableDiffusion3Pipeline,
    concept_dataset_dir: str,
    ref_mode: str = "first",
    ref_idx: int = 0,
    seed: int = 42
) -> torch.Tensor:
    """지정된 모드에 따라 레퍼런스 이미지(들)의 Latent 텐서를 추출/결합"""
    valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(concept_dataset_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"레퍼런스 이미지를 찾을 수 없습니다: {concept_dataset_dir}")

    if ref_mode == "first":
        selected_path = image_paths[0]
        print(f"  [Ref Mode: first] {os.path.basename(selected_path)} 선택됨")
        img = Image.open(selected_path)
        return encode_image_to_sd3_latent(pipe, img, seed=seed)

    elif ref_mode == "index":
        selected_idx = min(ref_idx, len(image_paths) - 1)
        selected_path = image_paths[selected_idx]
        print(f"  [Ref Mode: index {selected_idx}] {os.path.basename(selected_path)} 선택됨")
        img = Image.open(selected_path)
        return encode_image_to_sd3_latent(pipe, img, seed=seed)

    elif ref_mode == "avg":
        print(f"  [Ref Mode: avg] 총 {len(image_paths)}장 레퍼런스 Latent 앙상블 평균 계산 중...")
        latents = []
        for p in image_paths:
            img = Image.open(p)
            lat = encode_image_to_sd3_latent(pipe, img, seed=seed)
            latents.append(lat)
        avg_latent = torch.stack(latents, dim=0).mean(dim=0)
        return avg_latent

    else:
        raise ValueError(f"지원하지 않는 ref_mode입니다: {ref_mode}")


# ==============================================================================
# 3. Pipeline Loader
# ==============================================================================

def load_sd3_pipeline(device_type: str = "cuda") -> Tuple[StableDiffusion3Pipeline, dict]:
    """SD3.5-medium 파이프라인 로드"""
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    print(f"📦 SD3.5-medium 로딩 중 ({model_id})...")

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("✓ HF_TOKEN 감지됨")

    dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else (
        torch.float16 if device_type == "cuda" else torch.float32
    )

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        text_encoder_3=None,  # T5-XXL 생략으로 메모리 절약
        tokenizer_3=None,
        torch_dtype=dtype,
        token=hf_token
    )

    # A100 등 고용량 VRAM(>20GB)에서는 직접 cuda 로드하여 속도 극대화
    if device_type == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_vram >= 24:
            print(f"✓ VRAM {total_vram:.1f}GB 감지: 전체 파이프라인을 GPU로 직접 로드 (고속 모드)")
            pipeline = pipeline.to("cuda")
        else:
            print(f"✓ VRAM {total_vram:.1f}GB 감지: CPU offload 활성화")
            pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to("cpu")

    base_scheduler_config = dict(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(desc="Processing")
    print("✓ SD3.5-medium 파이프라인 준비 완료!")
    return pipeline, base_scheduler_config


# ==============================================================================
# 4. Core Inversion & Generation Function
# ==============================================================================

def run_inversion_and_generate(
    pipeline: StableDiffusion3Pipeline,
    base_scheduler_config: dict,
    concept: str,
    dataset_dir: str = "./dataset",
    prompts_dir: str = "./prompt",
    output_dir: str = "./generated",
    method: str = "rf",
    ref_mode: str = "first",
    ref_idx: int = 0,
    tau: float = 0.7,
    eta: float = 0.9,
    gamma: float = 0.5,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    seed: int = 42,
):
    if concept not in CLASS_PROMPT:
        print(f"[오류] 알 수 없는 서브젝트명입니다: {concept}")
        return

    class_word = CLASS_PROMPT[concept]
    concept_data_dir = os.path.join(dataset_dir, concept)
    prompt_file = os.path.join(prompts_dir, f"{concept}.txt")

    if not os.path.exists(prompt_file):
        print(f"[오류] 프롬프트 파일이 없습니다: {prompt_file}")
        return

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts_raw = [l.strip() for l in f.readlines() if l.strip()]

    concept_out_dir = os.path.join(output_dir, concept)
    os.makedirs(concept_out_dir, exist_ok=True)

    device = pipeline._execution_device
    print(f"\n========================================================")
    print(f"▶ [{concept}] Subject-driven {method.upper()}-Inversion 파이프라인 가동")
    print(f"  - Class Word: '{class_word}' | 레퍼런스 모드: {ref_mode}")
    print(f"  - Steps: {num_inference_steps} | CFG: {guidance_scale} | tau: {tau}, eta: {eta}")
    print(f"========================================================")

    # 1. Reference Latent 추출
    image_latent = get_reference_latents(
        pipe=pipeline,
        concept_dataset_dir=concept_data_dir,
        ref_mode=ref_mode,
        ref_idx=ref_idx,
        seed=seed
    )

    # 2. Inversion 단계 수행
    print(f"\n[Step 1/2] Latent Inversion 역추적 진행 중 (method: {method})...")
    if method == "rf":
        prior_reference = torch.randn(
            image_latent.shape,
            generator=torch.Generator(device=device).manual_seed(seed),
            device=image_latent.device,
            dtype=image_latent.dtype,
        )
        rf_inversion_scheduler = ControlledODEInversion.from_config(base_scheduler_config)
        rf_inversion_scheduler.set(prior_reference, tau=0.0, eta=gamma)
        pipeline.scheduler = rf_inversion_scheduler

        inverted_latent = pipeline(
            prompt="",  # null-text inversion
            num_inference_steps=num_inference_steps,
            height=512,
            width=512,
            guidance_scale=1.0,
            latents=image_latent,
            output_type="latent",
        ).images.detach()

    elif method == "euler":
        pipeline.scheduler = EulerInversion.from_config(base_scheduler_config)
        inverted_latent = pipeline(
            prompt=f"photo of a {class_word}",
            num_inference_steps=num_inference_steps,
            height=512,
            width=512,
            guidance_scale=1.0,
            latents=image_latent,
            output_type="latent",
        ).images.detach()

    else:
        raise ValueError(f"지원되지 않는 method입니다: {method}")

    print(f"✓ Inversion 완료! (Inverted Latent Mean: {inverted_latent.float().mean().item():.4f})")

    # 3. Controlled Generation 단계
    print(f"\n[Step 2/2] 10개 프롬프트 기반 맞춤 생성 시작...")
    if method == "rf":
        controlled_scheduler = ControlledODE.from_config(base_scheduler_config)
        controlled_scheduler.set(image_latent, tau=tau, eta=eta)
        pipeline.scheduler = controlled_scheduler
    elif method == "euler":
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_scheduler_config)

    for idx, raw_p in enumerate(prompts_raw):
        prompt_text = raw_p.replace("{}", class_word)
        out_path = os.path.join(concept_out_dir, f"{idx}.png")
        print(f"  [{idx}/9] 프롬프트: \"{prompt_text}\"")

        generator = torch.Generator(device=device).manual_seed(seed + idx)

        image = pipeline(
            prompt=prompt_text,
            negative_prompt="low quality, bad resolution, blurry, distorted, bad anatomy",
            num_inference_steps=num_inference_steps,
            height=512,
            width=512,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=inverted_latent,
        ).images[0]

        image.save(out_path)
        print(f"      -> 생성 및 저장 완료: {out_path}")

    # 원본 스케줄러로 복구
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_scheduler_config)
    print(f"✓ [{concept}] 10장 이미지 생성 완료 -> {concept_out_dir}")


# ==============================================================================
# 5. Evaluation Runner
# ==============================================================================

def run_evaluation(concept: str, dataset_dir: str = "./dataset", prompts_dir: str = "./prompt", generated_dir: str = "./generated"):
    print(f"\n📊 [{concept}] CLIP Evaluation 측정 중...")
    cmd = [
        sys.executable,
        "evaluation.py",
        "--dataset", dataset_dir,
        "--prompts", prompts_dir,
        "--concept", concept,
        "--images", os.path.join(generated_dir, concept)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(res.stdout)
        lines = res.stdout.strip().split("\n")
        t2i, i2i = None, None
        for l in lines:
            if "Text-to-Image Score:" in l:
                t2i = float(l.split(":")[1].strip())
            elif "Image-to-Image Score:" in l:
                i2i = float(l.split(":")[1].strip())
        return t2i, i2i
    except subprocess.CalledProcessError as e:
        print(f"[오류] Evaluation 실행 중 문제 발생:\n{e.stderr}")
        return None, None


# ==============================================================================
# 6. Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Subject-driven Inversion & Customization Pipeline")
    parser.add_argument("--concept", type=str, default="all", help="서브젝트명 ('all' 또는 특정 서브젝트)")
    parser.add_argument("--dataset", type=str, default="./dataset", help="레퍼런스 데이터셋 경로 (예: ./dataset 또는 ./augmentation)")
    parser.add_argument("--prompts", type=str, default="./prompt", help="프롬프트 폴더 경로")
    parser.add_argument("--output", type=str, default="./generated", help="생성 이미지 저장 경로")
    
    # Inversion 알고리즘 및 하이퍼파라미터
    parser.add_argument("--method", type=str, default="rf", choices=["rf", "euler"], help="Inversion 방법 (rf: Controlled ODE RF-Inversion, euler: Euler time-reversal)")
    parser.add_argument("--ref_mode", type=str, default="first", choices=["first", "avg", "index"], help="레퍼런스 이미지 선택 모드 (first: 0번, avg: 전체 Latent 평균 앙상블, index: 지정 인덱스)")
    parser.add_argument("--ref_idx", type=int, default=0, help="ref_mode가 index일 때 사용할 레퍼런스 번호")
    parser.add_argument("--tau", type=float, default=0.7, help="RF Controlled generation threshold tau (기본 0.7)")
    parser.add_argument("--eta", type=float, default=0.9, help="RF Controlled generation velocity weight eta (기본 0.9)")
    parser.add_argument("--gamma", type=float, default=0.5, help="RF Inversion prior weight gamma (기본 0.5)")
    
    # 일반 생성 파라미터
    parser.add_argument("--steps", type=int, default=28, help="인퍼런스 스텝 수 (기본 28)")
    parser.add_argument("--cfg", type=float, default=7.0, help="Guidance scale (기본 7.0)")
    parser.add_argument("--seed", type=int, default=42, help="시드값 (기본 42)")
    parser.add_argument("--no_eval", action="store_true", help="생성 후 evaluation 자동 평가 비활성화")
    parser.add_argument("--results_dir", type=str, default="./results", help="평가 결과 보고서 저장 폴더")

    args = parser.parse_args()

    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("   SD3.5 Subject-driven Inversion & Generation Pipeline")
    print("=" * 70)
    print(f"- 실행 Device: {device}")
    print(f"- 대상 서브젝트: {args.concept}")
    print(f"- 방법론: {args.method.upper()}-Inversion (Ref: {args.ref_mode})")
    print(f"- 데이터셋: {args.dataset} | 출력: {args.output}")
    print(f"- Steps: {args.steps} | CFG: {args.cfg} | Seed: {args.seed}")
    if args.method == "rf":
        print(f"- RF Hyperparameters: tau={args.tau}, eta={args.eta}, gamma={args.gamma}")

    pipeline, base_scheduler_config = load_sd3_pipeline(device_type=device)

    if args.concept == "all":
        target_concepts = list(CLASS_PROMPT.keys())
    else:
        target_concepts = [args.concept]

    eval_results = {}

    for concept in target_concepts:
        run_inversion_and_generate(
            pipeline=pipeline,
            base_scheduler_config=base_scheduler_config,
            concept=concept,
            dataset_dir=args.dataset,
            prompts_dir=args.prompts,
            output_dir=args.output,
            method=args.method,
            ref_mode=args.ref_mode,
            ref_idx=args.ref_idx,
            tau=args.tau,
            eta=args.eta,
            gamma=args.gamma,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            seed=args.seed
        )
        if not args.no_eval:
            t2i, i2i = run_evaluation(
                concept=concept,
                dataset_dir=args.dataset,
                prompts_dir=args.prompts,
                generated_dir=args.output
            )
            if t2i is not None and i2i is not None:
                eval_results[concept] = {"t2i": round(t2i, 4), "i2i": round(i2i, 4)}

    elapsed = time.time() - start_time

    # 전체 평가 요약 및 파일 저장
    if eval_results:
        os.makedirs(args.results_dir, exist_ok=True)
        avg_t2i = round(sum(v["t2i"] for v in eval_results.values()) / len(eval_results), 4)
        avg_i2i = round(sum(v["i2i"] for v in eval_results.values()) / len(eval_results), 4)

        summary_data = {
            "method": f"{args.method.upper()}-Inversion",
            "ref_mode": args.ref_mode,
            "hyperparameters": {
                "steps": args.steps,
                "cfg": args.cfg,
                "tau": args.tau if args.method == "rf" else None,
                "eta": args.eta if args.method == "rf" else None,
                "gamma": args.gamma if args.method == "rf" else None,
                "seed": args.seed,
            },
            "dataset": args.dataset,
            "elapsed_seconds": round(elapsed, 2),
            "average_scores": {
                "CLIP-T": avg_t2i,
                "CLIP-I": avg_i2i,
                "CLIP-Total": round(avg_t2i + avg_i2i, 4),
            },
            "per_concept_scores": eval_results
        }

        # JSON 저장
        json_path = os.path.join(args.results_dir, f"eval_{args.method}_{args.ref_mode}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        # Markdown 보고서 생성
        md_path = os.path.join(args.results_dir, f"EVALUATION_REPORT_{args.method.upper()}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 📊 Subject-driven {args.method.upper()}-Inversion Evaluation Report\n\n")
            f.write(f"- **실행 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **소요 시간**: {elapsed:.1f}초 ({elapsed/60:.1f}분)\n")
            f.write(f"- **방법론**: `{args.method.upper()}-Inversion` (Ref Mode: `{args.ref_mode}`)\n")
            f.write(f"- **데이터셋 경로**: `{args.dataset}`\n")
            f.write(f"- **하이퍼파라미터**: Steps={args.steps}, CFG={args.cfg}, tau={args.tau}, eta={args.eta}, seed={args.seed}\n\n")
            f.write("## 1. 정량 평가 요약 (CLIP Scores)\n\n")
            f.write("| Concept | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined (T+I) |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            for c, scores in eval_results.items():
                tot = round(scores['t2i'] + scores['i2i'], 4)
                f.write(f"| `{c}` | **{scores['t2i']:.4f}** | **{scores['i2i']:.4f}** | {tot:.4f} |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            f.write(f"| **전체 평균 (TOTAL AVG)** | **{avg_t2i:.4f}** | **{avg_i2i:.4f}** | **{round(avg_t2i + avg_i2i, 4):.4f}** |\n\n")
            f.write("> 💡 **발표 자료 팁**: ProjectOverview 요구사항에 따라 서브젝트별 10개 값 + 전체 평균 1개 = 총 22개 수치로 정리되어 있습니다.\n")

        print("\n" + "=" * 70)
        print("                  📈 전체 평가 결과 요약 (CLIP Scores)")
        print("=" * 70)
        print(f"{'Concept':<25} | {'Text-to-Image (CLIP-T)':<22} | {'Image-to-Image (CLIP-I)':<22}")
        print("-" * 75)
        for c, scores in eval_results.items():
            print(f"{c:<25} | {scores['t2i']:<22.4f} | {scores['i2i']:<22.4f}")
        print("-" * 75)
        print(f"{'TOTAL AVERAGE':<25} | {avg_t2i:<22.4f} | {avg_i2i:<22.4f}")
        print("=" * 70)
        print(f"✓ 결과 보고서 저장 완료: {json_path}, {md_path}")

    print("\n🎉 모든 Inversion 파이프라인 및 평가 과정이 완료되었습니다!")


if __name__ == "__main__":
    main()
