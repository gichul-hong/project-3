"""
SD3.5 LoRA + RF-Inversion Hybrid Generation & Evaluation Pipeline (Iter 4/6/7)
-----------------------------------------------------------------------------
[아이디어 40% 핵심 차별화 기법]
LoRA 파인튜닝 모델(높은 CLIP-T) + Controlled ODE RF-Inversion(높은 CLIP-I)을 결합하여
프롬프트 준수력과 객체 정체성을 동시에 극대화하는 하이브리드 파이프라인.

확장 기능:
1. Multi-reference Latent Averaging (nobg / raw ensemble).
2. Adaptive eta schedule (Cosine / Power decay) for smooth guidance transition.
3. FlowMatch Heun 2nd-order ODE Solver & 50-step inference.
4. Concept-specific negative prompts.
5. T5-XXL 텍스트 인코더 연동.

사용법:
    1) Exp-06 (LoRA HQ + Adaptive eta + Multi-reference avg):
       python generate_hybrid.py --concept all --checkpoints_dir ./checkpoints/exp05_lora_hq --ref_mode avg --eta_schedule adaptive --output ./experiments/06_hybrid_adaptive

    2) Exp-07 (Exp-06 + Heun 50 steps + Custom Negative Prompt):
       python generate_hybrid.py --concept all --checkpoints_dir ./checkpoints/exp05_lora_hq --ref_mode avg --eta_schedule adaptive --scheduler heun --steps 50 --custom_neg --output ./experiments/07_heun_custom_neg
"""

import argparse
import glob
import json
import math
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
    FlowMatchHeunDiscreteScheduler,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput,
)
from diffusers.schedulers.scheduling_flow_match_heun_discrete import (
    FlowMatchHeunDiscreteSchedulerOutput,
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

DEFAULT_NEGATIVE_PROMPTS = {
    "actionfigure_2": "human skin, real human face, photographic skin texture, blurry, distorted joints, bad anatomy, deformed plastic",
    "decoritems_woodenpot": "plastic, metallic, glossy, blurry, low resolution, deformed opening, distorted shape",
    "furniture_sofa2": "deformed shape, wrong color, bad perspective, wooden chair, bed, messy fabric, blurry, distorted legs",
    "instrument_music2": "piano, drums, distorted guitar neck, missing strings, extra headstock, blurry, bad anatomy",
    "luggage_backpack1": "handbag, plastic bag, distorted straps, deformed zipper, blurry, bad texture",
    "person_3": "distorted face, blurry eyes, extra limbs, bad anatomy, deformed fingers, low resolution, cartoon, 3d render, anime",
    "pet_cat5": "dog, ugly fur, distorted whiskers, extra paws, deformed eyes, blurry, bad anatomy",
    "scene_waterfall": "dry rocks, static water, cartoon, low resolution, distorted horizon, messy textures",
    "transport_tank": "toy, plastic miniature, cartoon, low resolution, blurry, deformed armor, civilian car",
    "wearable_jacket1": "shirt, hoodie, distorted collar, missing zipper, low resolution, blurry, deformed cloth",
}

CONCEPT_PROMPT_ENHANCERS = {
    "person_3": "detailed facial features, realistic eyes and skin texture, sharp portrait focus",
    "transport_tank": "heavy armor plating, realistic metallic surface, highly detailed tracks",
    "pet_cat5": "fine fur texture, clear feline eyes, realistic whiskers",
    "actionfigure_2": "smooth plastic texture, intricate figurine joints, studio lighting",
    "decoritems_woodenpot": "authentic wood grain texture, handcrafted details",
    "furniture_sofa2": "rich fabric texture, realistic cushions, clean geometry",
    "instrument_music2": "detailed guitar strings, wooden lacquer finish, clear frets",
    "luggage_backpack1": "detailed fabric stitching, realistic zippers and straps",
    "scene_waterfall": "flowing water dynamics, mist spray, highly detailed rock face",
    "wearable_jacket1": "detailed leather and fabric seams, realistic folds and zipper",
}

SUBJECT_ROUTING_PARAMS = {
    # Rigid / Object / Artifact concepts: structure preservation is key
    "furniture_sofa2": {"tau": 0.75, "eta": 0.90},
    "decoritems_woodenpot": {"tau": 0.75, "eta": 0.90},
    "instrument_music2": {"tau": 0.75, "eta": 0.90},
    "transport_tank": {"tau": 0.75, "eta": 0.90},
    "luggage_backpack1": {"tau": 0.75, "eta": 0.85},
    "wearable_jacket1": {"tau": 0.75, "eta": 0.85},
    # Flexible / Dynamic / Biological / Scene concepts: prompt/pose flexibility is key
    "person_3": {"tau": 0.60, "eta": 0.70},
    "pet_cat5": {"tau": 0.60, "eta": 0.70},
    "actionfigure_2": {"tau": 0.65, "eta": 0.75},
    "scene_waterfall": {"tau": 0.65, "eta": 0.75},
}


# ==============================================================================
# 1. Controlled ODE Schedulers (Euler & Heun with Adaptive Eta)
# ==============================================================================

class ControlledODE(FlowMatchEulerDiscreteScheduler):
    """원본 레퍼런스 latent 방향의 conditional velocity를 혼합하는 생성 Scheduler"""

    def set(
        self,
        reference: torch.Tensor,
        tau: float = 0.7,
        eta: float = 0.8,
        schedule: str = "adaptive",
        alpha: float = 1.5,
    ):
        self.reference = reference
        self.tau = float(tau)
        self.eta = float(eta)
        self.schedule = schedule
        self.alpha = float(alpha)
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
        sigma_val = sigma.item()

        if sigma_val > self.tau:
            if self.schedule == "adaptive":
                progress = (sigma_val - self.tau) / max(1.0 - self.tau, 1e-6)
                current_eta = self.eta * (progress ** self.alpha)
            elif self.schedule == "cosine":
                progress = (sigma_val - self.tau) / max(1.0 - self.tau, 1e-6)
                current_eta = self.eta * math.sin(progress * math.pi / 2.0)
            else:
                current_eta = self.eta
        else:
            current_eta = 0.0

        controlled_velocity = model_output + current_eta * (conditional_velocity - model_output)
        prev_sample = sample_fp32 + (sigma_next - sigma) * controlled_velocity
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)


class ControlledHeunODE(FlowMatchHeunDiscreteScheduler):
    """FlowMatch Heun 2nd-order Scheduler에 Controlled ODE Velocity를 적용"""

    def set(
        self,
        reference: torch.Tensor,
        tau: float = 0.7,
        eta: float = 0.8,
        schedule: str = "adaptive",
        alpha: float = 1.5,
    ):
        self.reference = reference
        self.tau = float(tau)
        self.eta = float(eta)
        self.schedule = schedule
        self.alpha = float(alpha)
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
        sigma_val = sigma.item()

        if sigma_val > self.tau:
            if self.schedule == "adaptive":
                progress = (sigma_val - self.tau) / max(1.0 - self.tau, 1e-6)
                current_eta = self.eta * (progress ** self.alpha)
            else:
                current_eta = self.eta
        else:
            current_eta = 0.0

        controlled_velocity = model_output + current_eta * (conditional_velocity - model_output)
        prev_sample = sample_fp32 + (sigma_next - sigma) * controlled_velocity
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return FlowMatchHeunDiscreteSchedulerOutput(prev_sample=prev_sample)


class ControlledODEInversion(ControlledODE):
    """Sampled prior (Noise) 방향의 conditional velocity를 보정하는 역변환 Scheduler"""

    def set(self, reference: torch.Tensor, tau: float = 0.0, eta: float = 0.5):
        return super().set(reference, tau=tau, eta=eta, schedule="step")

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
    concept: str,
    dataset_dir: str = "./dataset",
    aug_dir: str = "./augmentation",
    ref_mode: str = "avg",
    seed: int = 42
) -> torch.Tensor:
    concept_dir = os.path.join(dataset_dir, concept)
    concept_aug_dir = os.path.join(aug_dir, concept)
    
    valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(concept_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"레퍼런스 이미지를 찾을 수 없습니다: {concept_dir}")

    if ref_mode == "avg":
        # Raw dataset + nobg augmentation images 앙상블
        aug_nobg_paths = []
        if os.path.exists(concept_aug_dir):
            for ext in valid_exts:
                aug_nobg_paths.extend(glob.glob(os.path.join(concept_aug_dir, f"*_nobg{ext.replace('*', '')}")))
        
        all_paths = image_paths + sorted(aug_nobg_paths)
        print(f"  [Ref Mode: avg] 총 {len(all_paths)}개 레퍼런스(원본+nobg) Latent Multi-ref 평균 연산...")
        latents = []
        for p in all_paths:
            img = Image.open(p)
            lat = encode_image_to_sd3_latent(pipe, img, seed=seed)
            latents.append(lat)
        return torch.stack(latents, dim=0).mean(dim=0)

    elif ref_mode == "nobg":
        # nobg 증강 이미지 우선 모드
        if os.path.exists(concept_aug_dir):
            nobg_files = glob.glob(os.path.join(concept_aug_dir, "*_nobg.png"))
            if nobg_files:
                selected_path = sorted(nobg_files)[0]
                print(f"  [Ref Mode: nobg] {os.path.basename(selected_path)} 선택됨")
                img = Image.open(selected_path)
                return encode_image_to_sd3_latent(pipe, img, seed=seed)
        
        selected_path = image_paths[0]
        print(f"  [Ref Mode: nobg fallback -> first] {os.path.basename(selected_path)} 선택됨")
        img = Image.open(selected_path)
        return encode_image_to_sd3_latent(pipe, img, seed=seed)

    else:  # first
        selected_path = image_paths[0]
        print(f"  [Ref Mode: first] {os.path.basename(selected_path)} 선택됨")
        img = Image.open(selected_path)
        return encode_image_to_sd3_latent(pipe, img, seed=seed)


# ==============================================================================
# 3. Hybrid Inversion & Generation Pipeline
# ==============================================================================

def load_hybrid_pipeline(checkpoint_dir: str, device_type: str = "cuda", enable_t5: bool = True):
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    hf_token = os.getenv("HF_TOKEN")
    dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    t5_desc = "T5-XXL 포함" if enable_t5 else "T5 비활성화"
    print(f"📦 SD3.5-medium 로딩 중 ({model_id}, {t5_desc})...")

    if enable_t5:
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token
        )
    else:
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            text_encoder_3=None,
            tokenizer_3=None,
            torch_dtype=dtype,
            token=hf_token
        )

    base_scheduler_config = dict(pipeline.scheduler.config)

    if os.path.exists(checkpoint_dir):
        print(f"🔗 LoRA 가중치 로드 중: {checkpoint_dir}")
        try:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer,
                checkpoint_dir,
                torch_dtype=dtype
            )
            print("✓ PeftModel LoRA 가중치 주입 완료!")
        except Exception as e:
            print(f"⚠️ PeftModel 로드 대체 시도 (pipeline.load_lora_weights): {e}")
            pipeline.load_lora_weights(checkpoint_dir)
            print("✓ pipeline.load_lora_weights 완료!")
    else:
        print(f"[경고] LoRA 체크포인트를 찾을 수 없습니다: {checkpoint_dir}. Base 모델로 실행합니다.")

    if device_type == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_vram >= 24:
            pipeline = pipeline.to("cuda")
        else:
            pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to("cpu")

    return pipeline, base_scheduler_config


def run_hybrid_for_concept(
    pipeline: StableDiffusion3Pipeline,
    base_scheduler_config: dict,
    concept: str,
    dataset_dir: str = "./dataset",
    aug_dir: str = "./augmentation",
    prompts_dir: str = "./prompt",
    output_dir: str = "./experiments/06_hybrid_adaptive",
    instance_token: str = "sks",
    ref_mode: str = "avg",
    eta_schedule: str = "adaptive",
    scheduler_type: str = "euler",
    tau: float = 0.7,
    eta: float = 0.8,
    gamma: float = 0.5,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    use_concept_negative: bool = True,
    enhance_prompts: bool = False,
    seed: int = 42,
):
    if concept not in CLASS_PROMPT:
        print(f"[오류] 알 수 없는 서브젝트명: {concept}")
        return

    class_word = CLASS_PROMPT[concept]
    prompt_file = os.path.join(prompts_dir, f"{concept}.txt")

    if not os.path.exists(prompt_file):
        print(f"[오류] 프롬프트 파일 없음: {prompt_file}")
        return

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts_raw = [l.strip() for l in f.readlines() if l.strip()]

    concept_out_dir = os.path.join(output_dir, concept)
    os.makedirs(concept_out_dir, exist_ok=True)

    device = pipeline._execution_device
    token_word = f"{instance_token} {class_word}"

    if use_concept_negative and concept in DEFAULT_NEGATIVE_PROMPTS:
        neg_prompt = DEFAULT_NEGATIVE_PROMPTS[concept]
    else:
        neg_prompt = "low quality, bad resolution, blurry, distorted, bad anatomy"

    print(f"\n▶ [{concept}] LoRA + Controlled ODE Hybrid 파이프라인 시작 (Class: '{class_word}', Scheduler: '{scheduler_type}')")

    # 1. 레퍼런스 Latent 추출
    ref_latent = get_reference_latent(
        pipe=pipeline,
        concept=concept,
        dataset_dir=dataset_dir,
        aug_dir=aug_dir,
        ref_mode=ref_mode,
        seed=seed
    ).to(device=device, dtype=pipeline.vae.dtype)

    # 2. Controlled ODE Inversion (Sampled Prior Latent 역추적)
    print("  [Step 1] Inversion: 레퍼런스 Latent -> Inverted Noise Latent 계산 중...")
    inversion_scheduler = ControlledODEInversion.from_config(base_scheduler_config)
    inversion_scheduler.set(
        reference=torch.randn_like(ref_latent, generator=torch.Generator(device=device).manual_seed(seed)),
        tau=0.0,
        eta=gamma
    )
    pipeline.scheduler = inversion_scheduler
    pipeline.set_progress_bar_config(desc="Inversion")

    with torch.no_grad():
        inv_out = pipeline(
            prompt="",
            guidance_scale=1.0,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            latents=ref_latent,
        )
        inverted_latent = inv_out.images.clone()

    # 3. 10개 프롬프트 순차 생성 (Controlled ODE Generation)
    print(f"  [Step 2] Generation: 10개 테스트 프롬프트 이미지 생성 (tau={tau}, eta={eta}, schedule={eta_schedule})...")

    for idx, raw_p in enumerate(prompts_raw):
        prompt_text = raw_p.replace("{}", token_word)
        if enhance_prompts and concept in CONCEPT_PROMPT_ENHANCERS:
            enhancer = CONCEPT_PROMPT_ENHANCERS[concept]
            prompt_text = f"{prompt_text}, {enhancer}"
        out_path = os.path.join(concept_out_dir, f"{idx}.png")
        print(f"    [{idx}/9] 프롬프트: \"{prompt_text}\"")

        # Scheduler 재설정
        if scheduler_type == "heun":
            gen_scheduler = ControlledHeunODE.from_config(base_scheduler_config)
        else:
            gen_scheduler = ControlledODE.from_config(base_scheduler_config)

        gen_scheduler.set(
            reference=ref_latent,
            tau=tau,
            eta=eta,
            schedule=eta_schedule
        )
        pipeline.scheduler = gen_scheduler
        pipeline.set_progress_bar_config(desc=f"Generating [{idx}/9]")

        generator = torch.Generator(device=device).manual_seed(seed + idx)

        with torch.no_grad():
            gen_img = pipeline(
                prompt=prompt_text,
                negative_prompt=neg_prompt,
                num_inference_steps=num_inference_steps,
                height=512,
                width=512,
                guidance_scale=guidance_scale,
                latents=inverted_latent.clone(),
                generator=generator
            ).images[0]

        gen_img.save(out_path)
        print(f"        -> 저장 완료: {out_path}")

    print(f"✓ [{concept}] 10장 생성 완료 -> {concept_out_dir}")


def run_evaluation(concept: str, dataset_dir: str = "./dataset", prompts_dir: str = "./prompt", generated_dir: str = "./experiments/06_hybrid_adaptive"):
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


def main():
    parser = argparse.ArgumentParser(description="SD3.5 LoRA + RF-Inversion Hybrid Generation & Evaluation")
    parser.add_argument("--concept", type=str, default="actionfigure_2", help="서브젝트명 ('all' 또는 특정 서브젝트)")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="학습된 LoRA 체크포인트 디렉토리")
    parser.add_argument("--exp_name", type=str, default="", help="실험 세부 폴더 (지정 시 ./checkpoints/<exp_name>/ 에서 체크포인트 로드)")
    parser.add_argument("--dataset", type=str, default="./dataset", help="레퍼런스 이미지 데이터셋 경로")
    parser.add_argument("--aug_dir", type=str, default="./augmentation", help="증강 데이터셋 경로")
    parser.add_argument("--prompts", type=str, default="./prompt", help="프롬프트 폴더 경로")
    parser.add_argument("--output", type=str, default="./experiments/06_hybrid_adaptive", help="생성 결과 및 보고서 저장 폴더")
    parser.add_argument("--instance_token", type=str, default="sks", help="인스턴스 토큰")
    parser.add_argument("--ref_mode", type=str, default="avg", choices=["avg", "nobg", "first"], help="레퍼런스 이미지 선택 모드")
    parser.add_argument("--eta_schedule", type=str, default="adaptive", choices=["adaptive", "cosine", "step"], help="Adaptive eta 스케줄")
    parser.add_argument("--scheduler", type=str, default="euler", choices=["euler", "heun"], help="ODE Solver 스케줄러")
    parser.add_argument("--tau", type=float, default=0.7, help="Controlled generation threshold tau")
    parser.add_argument("--eta", type=float, default=0.8, help="Controlled generation velocity weight eta")
    parser.add_argument("--gamma", type=float, default=0.5, help="Controlled inversion prior weight gamma")
    parser.add_argument("--steps", type=int, default=28, help="인퍼런스 스텝 수")
    parser.add_argument("--cfg", type=float, default=7.0, help="CFG Scale")
    parser.add_argument("--enable_t5", action="store_true", default=True, help="T5-XXL 활성화")
    parser.add_argument("--no_t5", action="store_false", dest="enable_t5", help="T5-XXL 비활성화")
    parser.add_argument("--custom_neg", action="store_true", default=True, help="서브젝트별 맞춤 negative prompt 적용")
    parser.add_argument("--subject_routing", action="store_true", help="서브젝트 특성(Rigid vs Flexible) 기반 tau/eta 동적 라우팅")
    parser.add_argument("--enhance_prompts", action="store_true", help="서브젝트별 생성 프롬프트 고화질 디테일 수식어 자동 강화")
    parser.add_argument("--seed", type=int, default=42, help="시드값")
    parser.add_argument("--no_eval", action="store_true", help="자동 평가 비활성화")

    args = parser.parse_args()

    actual_checkpoints_dir = os.path.join(args.checkpoints_dir, args.exp_name) if args.exp_name else args.checkpoints_dir

    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)

    if args.concept == "all":
        target_concepts = list(CLASS_PROMPT.keys())
    else:
        target_concepts = [args.concept]

    eval_results = {}

    for concept in target_concepts:
        ckpt_dir = os.path.join(actual_checkpoints_dir, f"lora_{concept}")
        pipeline, base_config = load_hybrid_pipeline(checkpoint_dir=ckpt_dir, device_type=device, enable_t5=args.enable_t5)

        cur_tau = args.tau
        cur_eta = args.eta
        if args.subject_routing and concept in SUBJECT_ROUTING_PARAMS:
            cur_tau = SUBJECT_ROUTING_PARAMS[concept]["tau"]
            cur_eta = SUBJECT_ROUTING_PARAMS[concept]["eta"]
            print(f"  🧭 [Subject Routing] {concept}: tau={cur_tau}, eta={cur_eta} 동적 적용")

        run_hybrid_for_concept(
            pipeline=pipeline,
            base_scheduler_config=base_config,
            concept=concept,
            dataset_dir=args.dataset,
            aug_dir=args.aug_dir,
            prompts_dir=args.prompts,
            output_dir=args.output,
            instance_token=args.instance_token,
            ref_mode=args.ref_mode,
            eta_schedule=args.eta_schedule,
            scheduler_type=args.scheduler,
            tau=cur_tau,
            eta=cur_eta,
            gamma=args.gamma,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            use_concept_negative=args.custom_neg,
            enhance_prompts=args.enhance_prompts,
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

    # 전체 평가 요약 및 파일 저장
    if eval_results:
        out_json_path = os.path.join(args.output, "eval_summary.json")
        out_md_path = os.path.join(args.output, "EVALUATION_REPORT.md")

        all_scores = {}
        if os.path.exists(out_json_path):
            try:
                with open(out_json_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                all_scores.update(old_data.get("per_concept_scores", {}))
            except Exception:
                pass
        all_scores.update(eval_results)

        avg_t2i = round(sum(v["t2i"] for v in all_scores.values()) / len(all_scores), 4)
        avg_i2i = round(sum(v["i2i"] for v in all_scores.values()) / len(all_scores), 4)

        summary_data = {
            "method": "SD3.5 LoRA + Controlled ODE Hybrid",
            "instance_token": args.instance_token,
            "hyperparameters": {
                "steps": args.steps,
                "cfg": args.cfg,
                "tau": args.tau,
                "eta": args.eta,
                "gamma": args.gamma,
                "ref_mode": args.ref_mode,
                "eta_schedule": args.eta_schedule,
                "scheduler": args.scheduler,
                "custom_neg": args.custom_neg,
                "enable_t5": args.enable_t5,
                "seed": args.seed,
            },
            "elapsed_seconds": round(elapsed, 2),
            "average_scores": {
                "CLIP-T": avg_t2i,
                "CLIP-I": avg_i2i,
                "CLIP-Total": round(avg_t2i + avg_i2i, 4),
            },
            "per_concept_scores": all_scores
        }

        # JSON & Markdown 동시 저장
        out_json_path = os.path.join(args.output, "eval_summary.json")
        out_md_path = os.path.join(args.output, "EVALUATION_REPORT.md")

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        report_content = (
            f"# 📊 Subject-driven LoRA + RF-Inversion Hybrid Evaluation Report\n\n"
            f"- **실행 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- **소요 시간**: {elapsed:.1f}초 ({elapsed/60:.1f}분)\n"
            f"- **방법론**: `LoRA Fine-Tuning + Controlled ODE Inversion Hybrid ({args.eta_schedule} eta, {args.ref_mode} ref)`\n"
            f"- **하이퍼파라미터**: Steps={args.steps} ({args.scheduler}), CFG={args.cfg}, tau={args.tau}, eta={args.eta}, Token='{args.instance_token}'\n\n"
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

        # 재실행 가이드용 README.md 생성
        exp_readme_path = os.path.join(args.output, "README.md")
        readme_content = f"""# 🧪 Experiment: {os.path.basename(args.output)}

## 1. 실험 개요 및 방법론
- **방법론**: `SD3.5 LoRA + Controlled ODE RF-Inversion Hybrid`
- **스케줄러**: `{args.scheduler}` ({args.steps} steps)
- **Controlled ODE 설정**: tau={args.tau}, eta={args.eta}, gamma={args.gamma}, eta_schedule={args.eta_schedule}
- **Reference 모드**: `{args.ref_mode}` (Multi-reference Latent Ensemble)
- **T5-XXL 텍스트 인코더**: {'활성화' if args.enable_t5 else '비활성화'}
- **Custom Negative Prompt**: {'적용' if args.custom_neg else '미적용'}

## 2. 재실행(Reproduction) 명령어
```bash
python generate_hybrid.py \\
    --concept all \\
    --checkpoints_dir {args.checkpoints_dir} \\
    --output {args.output} \\
    --ref_mode {args.ref_mode} \\
    --eta_schedule {args.eta_schedule} \\
    --scheduler {args.scheduler} \\
    --tau {args.tau} \\
    --eta {args.eta} \\
    --steps {args.steps} \\
    --enable_t5 \\
    --custom_neg
```

## 3. 평가 점수 요약
- **Text-to-Image (CLIP-T)**: **{avg_t2i:.4f}**
- **Image-to-Image (CLIP-I)**: **{avg_i2i:.4f}**
- **Total Combined (T+I)**: **{round(avg_t2i + avg_i2i, 4):.4f}**

> 📌 상세 10개 서브젝트별 22개 점수표: [`EVALUATION_REPORT.md`](file://{os.path.abspath(out_md_path)})
"""
        with open(exp_readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

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

    print("\n🎉 모든 Hybrid 생성 및 평가 과정이 완료되었습니다!")


if __name__ == "__main__":
    main()
