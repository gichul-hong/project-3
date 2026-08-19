"""
SD3.5 LoRA + RF-Inversion Hybrid Generation & Evaluation Pipeline (Iter 4)
-------------------------------------------------------------------------
[아이디어 40% 핵심 차별화 기법]
LoRA 파인튜닝 모델(높은 CLIP-T) + Controlled ODE RF-Inversion(높은 CLIP-I)을 결합하여
프롬프트 준수력과 객체 정체성을 동시에 극대화하는 하이브리드 파이프라인.

특징:
1. 학습된 LoRA 가중치를 SD3.5 백본에 주입.
2. 레퍼런스 이미지(또는 nobg 증강 이미지)를 VAE Latent로 인코딩.
3. LoRA 모델 기반 Controlled ODE Inversion (Sampled Prior 역추적).
4. LoRA 모델 + Controlled ODE Generation (Reference Latent 방향 Velocity 혼합).
5. 100장 이미지 자동 생성 및 CLIP-T / CLIP-I 종합 평가 및 리포트 저장.

사용법:
    1) 샘플 서브젝트 빠른 테스트:
       python generate_hybrid.py --concept actionfigure_2 --tau 0.7 --eta 0.8

    2) 전체 10개 서브젝트 일괄 실행 (Exp-04):
       python generate_hybrid.py --concept all --output ./experiments/04_lora_rf_hybrid
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

from diffusers import (
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput,
)
from peft import PeftModel

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
# 1. Controlled ODE Schedulers (Flow Matching RF-Inversion)
# ==============================================================================

class ControlledODE(FlowMatchEulerDiscreteScheduler):
    """원본 레퍼런스 latent 방향의 conditional velocity를 혼합하는 생성 Scheduler"""

    def set(self, reference: torch.Tensor, tau: float = 0.7, eta: float = 0.8):
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
# 2. VAE Encoding & Reference Helpers
# ==============================================================================

@torch.no_grad()
def encode_image_to_sd3_latent(pipe: StableDiffusion3Pipeline, image: Image.Image, seed: int = 0) -> torch.Tensor:
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


def get_reference_latent(
    pipe: StableDiffusion3Pipeline,
    concept_dir: str,
    ref_mode: str = "first",
    seed: int = 42
) -> torch.Tensor:
    valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(concept_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"레퍼런스 이미지를 찾을 수 없습니다: {concept_dir}")

    # nobg 이미지 우선 모드
    if ref_mode == "nobg":
        nobg_candidates = [p for p in image_paths if "nobg" in os.path.basename(p)]
        selected_path = nobg_candidates[0] if nobg_candidates else image_paths[0]
        print(f"  [Ref Mode: nobg] {os.path.basename(selected_path)} 선택됨")
        img = Image.open(selected_path)
        return encode_image_to_sd3_latent(pipe, img, seed=seed)

    elif ref_mode == "avg":
        print(f"  [Ref Mode: avg] 총 {len(image_paths)}장 레퍼런스 Latent 앙상블 계산 중...")
        latents = []
        for p in image_paths:
            img = Image.open(p)
            lat = encode_image_to_sd3_latent(pipe, img, seed=seed)
            latents.append(lat)
        return torch.stack(latents, dim=0).mean(dim=0)

    else:  # first
        selected_path = image_paths[0]
        print(f"  [Ref Mode: first] {os.path.basename(selected_path)} 선택됨")
        img = Image.open(selected_path)
        return encode_image_to_sd3_latent(pipe, img, seed=seed)


# ==============================================================================
# 3. Hybrid Pipeline Loader & Execution
# ==============================================================================

def load_hybrid_pipeline(checkpoint_dir: str, device_type: str = "cuda"):
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    hf_token = os.getenv("HF_TOKEN")
    dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    print(f"📦 SD3.5-medium 로딩 중 ({model_id})...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=dtype,
        token=hf_token
    )

    if os.path.exists(checkpoint_dir):
        print(f"🔗 LoRA 가중치 주입: {checkpoint_dir}")
        try:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer,
                checkpoint_dir,
                torch_dtype=dtype
            )
            print("✓ PeftModel LoRA 가중치 주입 완료!")
        except Exception as e:
            pipeline.load_lora_weights(checkpoint_dir)
            print(f"✓ pipeline.load_lora_weights 완료! ({e})")
    else:
        print(f"⚠️ LoRA 체크포인트 없음: {checkpoint_dir}. Base 모델로 실행합니다.")

    if device_type == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_vram >= 24:
            pipeline = pipeline.to("cuda")
        else:
            pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to("cpu")

    base_scheduler_config = dict(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(desc="Hybrid Inverting & Generating")
    return pipeline, base_scheduler_config


def run_hybrid_for_concept(
    pipeline: StableDiffusion3Pipeline,
    base_scheduler_config: dict,
    concept: str,
    dataset_dir: str = "./dataset",
    prompts_dir: str = "./prompt",
    output_dir: str = "./experiments/04_lora_rf_hybrid",
    instance_token: str = "sks",
    ref_mode: str = "first",
    tau: float = 0.7,
    eta: float = 0.8,
    gamma: float = 0.5,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    seed: int = 42,
):
    if concept not in CLASS_PROMPT:
        print(f"[오류] 알 수 없는 서브젝트명: {concept}")
        return

    class_word = CLASS_PROMPT[concept]
    concept_data_dir = os.path.join(dataset_dir, concept)
    prompt_file = os.path.join(prompts_dir, f"{concept}.txt")

    if not os.path.exists(prompt_file):
        print(f"[오류] 프롬프트 파일 없음: {prompt_file}")
        return

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts_raw = [l.strip() for l in f.readlines() if l.strip()]

    concept_out_dir = os.path.join(output_dir, concept)
    os.makedirs(concept_out_dir, exist_ok=True)

    device = pipeline._execution_device
    print(f"\n========================================================")
    print(f"▶ [{concept}] LoRA + Controlled ODE Hybrid 파이프라인 가동")
    print(f"  - Class: '{class_word}', Token: '{instance_token}' | Ref Mode: {ref_mode}")
    print(f"  - Steps: {num_inference_steps} | CFG: {guidance_scale} | tau: {tau}, eta: {eta}")
    print(f"========================================================")

    # 1. Reference Latent 추출
    image_latent = get_reference_latent(pipeline, concept_data_dir, ref_mode=ref_mode, seed=seed)

    # 2. Controlled ODE Inversion (LoRA 모델 기반 역추적)
    print(f"\n[Step 1/2] LoRA 결합 Controlled Inversion 수행 중...")
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

    print(f"✓ Hybrid Inversion 완료! (Inverted Latent Mean: {inverted_latent.float().mean().item():.4f})")

    # 3. LoRA + Controlled ODE Generation (프롬프트 적응 + 외형 보정)
    print(f"\n[Step 2/2] 10개 프롬프트 기반 하이브리드 생성 시작...")
    controlled_scheduler = ControlledODE.from_config(base_scheduler_config)
    controlled_scheduler.set(image_latent, tau=tau, eta=eta)
    pipeline.scheduler = controlled_scheduler

    for idx, raw_p in enumerate(prompts_raw):
        token_word = f"{instance_token} {class_word}"
        prompt_text = raw_p.replace("{}", token_word)
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

    # 원본 스케줄러 복구
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_scheduler_config)
    print(f"✓ [{concept}] 10장 이미지 생성 완료 -> {concept_out_dir}")


def run_evaluation(concept: str, dataset_dir: str = "./dataset", prompts_dir: str = "./prompt", generated_dir: str = "./experiments/04_lora_rf_hybrid"):
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
# 4. Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SD3.5 LoRA + RF-Inversion Hybrid Customization (Iter 4)")
    parser.add_argument("--concept", type=str, default="actionfigure_2", help="서브젝트명 ('all' 또는 특정 서브젝트)")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="학습된 LoRA 체크포인트 디렉토리")
    parser.add_argument("--dataset", type=str, default="./dataset", help="레퍼런스 이미지 데이터셋 경로")
    parser.add_argument("--prompts", type=str, default="./prompt", help="프롬프트 폴더 경로")
    parser.add_argument("--output", type=str, default="./experiments/04_lora_rf_hybrid", help="생성 결과 및 보고서 저장 폴더")
    parser.add_argument("--instance_token", type=str, default="sks", help="인스턴스 토큰")
    parser.add_argument("--ref_mode", type=str, default="first", choices=["first", "avg", "nobg"], help="레퍼런스 이미지 선택 모드")
    parser.add_argument("--tau", type=float, default=0.7, help="Controlled generation threshold tau")
    parser.add_argument("--eta", type=float, default=0.8, help="Controlled generation velocity weight eta")
    parser.add_argument("--gamma", type=float, default=0.5, help="Controlled inversion prior weight gamma")
    parser.add_argument("--steps", type=int, default=28, help="인퍼런스 스텝 수")
    parser.add_argument("--cfg", type=float, default=7.0, help="CFG Scale")
    parser.add_argument("--seed", type=int, default=42, help="시드값")
    parser.add_argument("--no_eval", action="store_true", help="자동 평가 비활성화")

    args = parser.parse_args()

    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)

    if args.concept == "all":
        target_concepts = list(CLASS_PROMPT.keys())
    else:
        target_concepts = [args.concept]

    eval_results = {}

    for concept in target_concepts:
        ckpt_dir = os.path.join(args.checkpoints_dir, f"lora_{concept}")
        pipeline, base_config = load_hybrid_pipeline(checkpoint_dir=ckpt_dir, device_type=device)

        run_hybrid_for_concept(
            pipeline=pipeline,
            base_scheduler_config=base_config,
            concept=concept,
            dataset_dir=args.dataset,
            prompts_dir=args.prompts,
            output_dir=args.output,
            instance_token=args.instance_token,
            ref_mode=args.ref_mode,
            tau=args.tau,
            eta=args.eta,
            gamma=args.gamma,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            seed=args.seed
        )

        # 메모리 해제
        del pipeline
        torch.cuda.empty_cache()

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

    # 종합 평가 요약 저장
    if eval_results:
        avg_t2i = round(sum(v["t2i"] for v in eval_results.values()) / len(eval_results), 4)
        avg_i2i = round(sum(v["i2i"] for v in eval_results.values()) / len(eval_results), 4)

        summary_data = {
            "method": "LoRA + RF-Inversion Hybrid",
            "instance_token": args.instance_token,
            "ref_mode": args.ref_mode,
            "hyperparameters": {
                "steps": args.steps,
                "cfg": args.cfg,
                "tau": args.tau,
                "eta": args.eta,
                "gamma": args.gamma,
                "seed": args.seed,
            },
            "elapsed_seconds": round(elapsed, 2),
            "average_scores": {
                "CLIP-T": avg_t2i,
                "CLIP-I": avg_i2i,
                "CLIP-Total": round(avg_t2i + avg_i2i, 4),
            },
            "per_concept_scores": eval_results
        }

        out_json_path = os.path.join(args.output, "eval_summary.json")
        out_md_path = os.path.join(args.output, "EVALUATION_REPORT.md")

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        report_content = (
            "# 📊 Subject-driven LoRA + RF-Inversion Hybrid Evaluation Report (Iter 4)\n\n"
            f"- **실행 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- **소요 시간**: {elapsed:.1f}초 ({elapsed/60:.1f}분)\n"
            f"- **방법론**: `LoRA Fine-Tuning + Controlled ODE Inversion Hybrid`\n"
            f"- **하이퍼파라미터**: Steps={args.steps}, CFG={args.cfg}, tau={args.tau}, eta={args.eta}, Token='{args.instance_token}'\n\n"
            "## 1. 정량 평가 요약 (CLIP Scores)\n\n"
            "| Concept | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined (T+I) |\n"
            "| :--- | :---: | :---: | :---: |\n"
        )
        for c, scores in eval_results.items():
            tot = round(scores['t2i'] + scores['i2i'], 4)
            report_content += f"| `{c}` | **{scores['t2i']:.4f}** | **{scores['i2i']:.4f}** | {tot:.4f} |\n"
        report_content += "| :--- | :---: | :---: | :---: |\n"
        report_content += f"| **전체 평균 (TOTAL AVG)** | **{avg_t2i:.4f}** | **{avg_i2i:.4f}** | **{round(avg_t2i + avg_i2i, 4):.4f}** |\n\n"

        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write(report_content)

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
        print(f"✓ 결과 보고서 저장 완료: {out_json_path}, {out_md_path}")

    print("\n🎉 하이브리드 파이프라인 생성 및 평가 과정이 완료되었습니다!")


if __name__ == "__main__":
    main()
