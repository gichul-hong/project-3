"""
SD3.5-Medium LoRA / DreamBooth Fine-Tuning Pipeline
---------------------------------------------------
A100 40GB 및 VRAM 효율성에 최적화된 Rectified Flow Matching LoRA 학습 스크립트.

특징:
1. VAE Latent & Text Embedding Pre-caching (T5-XXL 선택적 활성화):
   학습 전 증강 데이터셋의 모든 이미지와 캡션을 1회 인코딩하여 캐싱하므로 VRAM 소모 극소화 (8~12GB VRAM 사용) 및 초고속 학습.
2. SD3Transformer2DModel Attention 레이어에 PEFT LoRA 적용 (Rank 16 ~ 64+).
3. Rectified Flow Matching Loss (MSE on velocity field / Logit-Normal weighting).
4. Diffusers 표준 LoRA (.safetensors) 및 PEFT 체크포인트 이중 자동 저장.

사용법:
    1) 단일 서브젝트 학습 (Fast Test):
       python train_lora_sd3.py --concept actionfigure_2 --rank 64 --alpha 64 --steps 300 --enable_t5

    2) 전체 10개 서브젝트 고품질 학습 (Exp-05):
       python train_lora_sd3.py --concept all --rank 64 --alpha 64 --steps 500 --enable_t5 --exp_name exp05_lora_hq
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig, get_peft_model, PeftModel
from peft.utils import get_peft_model_state_dict

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
# 1. Dataset & Pre-caching Helper
# ==============================================================================

class AugmentedConceptDataset(Dataset):
    """증강 데이터셋(이미지 및 텍스트 캡션)을 로드하는 PyTorch Dataset"""

    def __init__(self, concept_dir: str, class_name: str, instance_token: str = "sks", image_size: int = 512):
        self.concept_dir = concept_dir
        self.class_name = class_name
        self.instance_token = instance_token
        self.image_size = image_size
        self.samples = []

        # metadata.jsonl이 있는 경우 우선 활용
        meta_path = os.path.join(concept_dir, "metadata.jsonl")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        img_path = os.path.join(concept_dir, data["file_name"])
                        if os.path.exists(img_path):
                            self.samples.append((img_path, data.get("text", f"a photo of {instance_token} {class_name}")))

        # metadata.jsonl이 없거나 비어있을 경우 이미지 & txt 파일 직접 스캔
        if not self.samples:
            valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
            img_paths = []
            for ext in valid_exts:
                img_paths.extend(glob.glob(os.path.join(concept_dir, ext)))
            for img_p in sorted(img_paths):
                txt_p = os.path.splitext(img_p)[0] + ".txt"
                if os.path.exists(txt_p):
                    with open(txt_p, "r", encoding="utf-8") as tf:
                        caption = tf.read().strip()
                else:
                    caption = f"a photo of {instance_token} {class_name}"
                self.samples.append((img_p, caption))

        if not self.samples:
            raise FileNotFoundError(f"데이터셋 이미지를 찾을 수 없습니다: {concept_dir}")

        # _nobg 배경 제거 이미지에 가중치를 부여하여 배경 얽힘 방지 및 피사체 집중 학습
        weighted_samples = []
        for img_p, caption in self.samples:
            weighted_samples.append((img_p, caption))
            if "nobg" in os.path.basename(img_p):
                weighted_samples.append((img_p, caption))  # nobg 2x 가중 반영
        self.samples = weighted_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        return image, caption


@torch.no_grad()
def precompute_embeddings_and_latents(
    pipe: StableDiffusion3Pipeline,
    dataset: AugmentedConceptDataset,
    device: torch.device,
    dtype: torch.dtype,
    enable_t5: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    """학습 속도와 VRAM 최적화를 위해 VAE Latent와 Text Embedding을 사전에 캐싱"""
    cached_data = []
    t5_status = "T5-XXL 포함" if enable_t5 else "CLIP-L/G 전용"
    print(f"⚡ VAE Latents 및 Text Embeddings 사전 캐싱 중 ({t5_status}, 총 {len(dataset)}개 샘플)...")

    for i in range(len(dataset)):
        image, caption = dataset[i]

        # 1. VAE Latent 인코딩
        image_tensor = pipe.image_processor.preprocess(image).to(device=device, dtype=pipe.vae.dtype)
        posterior = pipe.vae.encode(image_tensor).latent_dist
        raw_latent = posterior.sample()
        shift_factor = pipe.vae.config.shift_factor
        scaling_factor = pipe.vae.config.scaling_factor
        latent = (raw_latent - shift_factor) * scaling_factor
        latent = latent.to(dtype=dtype)

        # 2. Text Embeddings 추출
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=caption,
            prompt_2=caption,
            prompt_3=caption if enable_t5 else None,
            device=device,
            do_classifier_free_guidance=False
        )

        cached_data.append({
            "latent": latent.squeeze(0).cpu(),
            "prompt_embeds": prompt_embeds.squeeze(0).cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.squeeze(0).cpu(),
        })

    return cached_data


# ==============================================================================
# 2. Rectified Flow Matching LoRA Trainer
# ==============================================================================

def train_concept_lora(
    concept: str,
    dataset_dir: str = "./augmentation",
    output_base_dir: str = "./checkpoints",
    instance_token: str = "sks",
    r: int = 64,
    lora_alpha: int = 64,
    lr: float = 5e-5,
    max_train_steps: int = 1000,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 2,
    weighting_scheme: str = "flow_shift",
    enable_t5: bool = True,
    seed: int = 42,
):
    if concept not in CLASS_PROMPT:
        print(f"[오류] 알 수 없는 서브젝트명: {concept}")
        return None

    class_name = CLASS_PROMPT[concept]
    concept_dir = os.path.join(dataset_dir, concept)
    output_dir = os.path.join(output_base_dir, f"lora_{concept}")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"🚀 [{concept}] SD3.5 Rectified Flow LoRA 파인튜닝 시작")
    print(f"  - Class: '{class_name}', Token: '{instance_token}'")
    print(f"  - Rank: {r}, Alpha: {lora_alpha}, LR: {lr}, Max Steps: {max_train_steps}")
    print(f"  - T5 Text Encoder: {'활성화 (T5-XXL 4.7B)' if enable_t5 else '비활성화'}")
    print(f"  - Weighting Scheme: '{weighting_scheme}'")
    print(f"  - 저장 경로: {output_dir}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. 임시 파이프라인 로드로 VAE & Text Encoder 캐싱 수행
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    hf_token = os.getenv("HF_TOKEN")

    print(f"📦 VAE 및 텍스트 인코더 로드 중 (T5={'On' if enable_t5 else 'Off'})...")
    if enable_t5:
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token
        ).to(device)
    else:
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            text_encoder_3=None,
            tokenizer_3=None,
            torch_dtype=dtype,
            token=hf_token
        ).to(device)

    dataset = AugmentedConceptDataset(concept_dir, class_name, instance_token)
    cached_data = precompute_embeddings_and_latents(pipeline, dataset, device, dtype, enable_t5=enable_t5)

    # VAE와 Text Encoder 메모리 해제하여 VRAM 확보
    del pipeline.vae
    del pipeline.text_encoder
    del pipeline.text_encoder_2
    if hasattr(pipeline, "text_encoder_3") and pipeline.text_encoder_3 is not None:
        del pipeline.text_encoder_3
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    if hasattr(pipeline, "tokenizer_3") and pipeline.tokenizer_3 is not None:
        del pipeline.tokenizer_3
    del pipeline
    torch.cuda.empty_cache()

    # 2. SD3Transformer2DModel 백본 로드 및 LoRA 주입
    print("🧠 SD3Transformer2DModel 로드 및 LoRA 레이어 구성 중...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=dtype,
        token=hf_token
    ).to(device)

    # Gradient Checkpointing 활성화
    transformer.enable_gradient_checkpointing()

    # LoRA Config 설정 (MMDiT Attention Projections 대상)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, transformer.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_train_steps, eta_min=1e-6
    )

    # 3. 학습 루프 (Rectified Flow Matching Loss)
    transformer.train()
    progress_bar = tqdm(range(max_train_steps), desc=f"Training LoRA [{concept}]")
    step = 0
    num_samples = len(cached_data)

    while step < max_train_steps:
        # 미니배치 샘플링
        indices = torch.randint(0, num_samples, (batch_size,))
        batch_latents = torch.stack([cached_data[i]["latent"] for i in indices]).to(device=device, dtype=dtype)
        batch_prompt_embeds = torch.stack([cached_data[i]["prompt_embeds"] for i in indices]).to(device=device, dtype=dtype)
        batch_pooled_prompt_embeds = torch.stack([cached_data[i]["pooled_prompt_embeds"] for i in indices]).to(device=device, dtype=dtype)

        # Flow Matching 시간 변수 t 샘플링
        if weighting_scheme == "logit_normal":
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            ).to(device=device, dtype=dtype)
            sigmas = u
        else:
            # Shift 3.0 flow matching schedule
            u = torch.rand((batch_size,), device=device, dtype=dtype)
            shift = 3.0
            sigmas = shift * u / (1.0 + (shift - 1.0) * u)

        timesteps = sigmas * 1000.0  # scale to train timesteps (0~1000)

        # 노이즈 샘플링 및 Flow Matching Noisy Latents x_t 구성: x_t = (1-t) x_0 + t x_1
        noise = torch.randn_like(batch_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1, 1)
        noisy_latents = (1.0 - sigmas_expanded) * batch_latents + sigmas_expanded * noise

        # Target Velocity: v = noise - batch_latents (x_1 - x_0)
        target = noise - batch_latents

        # Model forward
        model_pred = transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=batch_prompt_embeds,
            pooled_projections=batch_pooled_prompt_embeds,
            return_dict=False,
        )[0]

        # Flow Matching Loss
        if weighting_scheme == "logit_normal":
            weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas_expanded).to(device=device, dtype=torch.float32)
            loss = torch.mean((weighting * (model_pred.float() - target.float()) ** 2))
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == max_train_steps:
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        step += 1
        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
            "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
        })

    # 4. LoRA 가중치 이중 저장 (PEFT format + Diffusers save_lora_weights format)
    print(f"\n💾 LoRA 가중치 저장 중: {output_dir}")
    transformer.save_pretrained(output_dir)

    try:
        peft_state_dict = get_peft_model_state_dict(transformer)
        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=output_dir,
            transformer_lora_layers=peft_state_dict,
        )
        print("✓ Diffusers 표준 LoRA safetensors 저장 완료!")
    except Exception as e:
        print(f"⚠️ Diffusers save_lora_weights 보조 저장 스킵: {e}")

    # 메타데이터 저장
    meta_info = {
        "concept": concept,
        "class_name": class_name,
        "instance_token": instance_token,
        "r": r,
        "lora_alpha": lora_alpha,
        "lr": lr,
        "max_train_steps": max_train_steps,
        "weighting_scheme": weighting_scheme,
        "enable_t5": enable_t5,
        "base_model": model_id,
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    print(f"✓ [{concept}] LoRA 학습 및 저장 완료 -> {output_dir}")

    # 메모리 정리
    del transformer
    del optimizer
    del lr_scheduler
    torch.cuda.empty_cache()

    return output_dir


# ==============================================================================
# 3. Main CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SD3.5 LoRA Fine-Tuning Pipeline")
    parser.add_argument("--concept", type=str, default="actionfigure_2", help="서브젝트명 ('all' 또는 특정 서브젝트)")
    parser.add_argument("--dataset", type=str, default="./augmentation", help="증강 데이터셋 경로")
    parser.add_argument("--exp_name", type=str, default="", help="실험 버전명 (지정 시 ./checkpoints/<exp_name>/ 에 저장)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="체크포인트 기본 저장 디렉토리")
    parser.add_argument("--instance_token", type=str, default="sks", help="인스턴스 고유 토큰")
    parser.add_argument("--rank", type=int, default=64, help="LoRA Rank (기본 64)")
    parser.add_argument("--alpha", type=int, default=64, help="LoRA Alpha (기본 64)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning Rate (기본 5e-5)")
    parser.add_argument("--steps", type=int, default=1000, help="학습 스텝 수 (기본 1000)")
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient Accumulation Steps")
    parser.add_argument("--weighting_scheme", type=str, default="flow_shift", choices=["flow_shift", "logit_normal"], help="Weighting Scheme")
    parser.add_argument("--enable_t5", action="store_true", default=True, help="T5-XXL 텍스트 인코더 활성화")
    parser.add_argument("--no_t5", action="store_false", dest="enable_t5", help="T5-XXL 비활성화")
    parser.add_argument("--seed", type=int, default=42, help="시드값")

    args = parser.parse_args()

    actual_output_dir = os.path.join(args.output_dir, args.exp_name) if args.exp_name else args.output_dir

    if args.concept == "all":
        target_concepts = list(CLASS_PROMPT.keys())
    else:
        target_concepts = [args.concept]

    start_time = time.time()
    for concept in target_concepts:
        train_concept_lora(
            concept=concept,
            dataset_dir=args.dataset,
            output_base_dir=actual_output_dir,
            instance_token=args.instance_token,
            r=args.rank,
            lora_alpha=args.alpha,
            lr=args.lr,
            max_train_steps=args.steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            weighting_scheme=args.weighting_scheme,
            enable_t5=args.enable_t5,
            seed=args.seed,
        )

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 모든 LoRA 파인튜닝 완료! (총 소요 시간: {total_time:.1f}초, {total_time/60:.1f}분)")
    print("=" * 70)


if __name__ == "__main__":
    main()
