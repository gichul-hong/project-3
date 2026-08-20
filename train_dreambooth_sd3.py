"""
True DreamBooth-LoRA Training Pipeline for SD3.5 (Rectified Flow Matching)
-------------------------------------------------------------------------
DreamBooth (Ruiz et al., CVPR 2023) 정석 구현:
1. Instance Dataset: "photo of a sks [class]" (사용자 제공 증강 데이터셋)
2. Class Prior Dataset: "photo of a [class]" (SD3.5 Base 생성 정규화 데이터셋)
3. Prior Preservation Loss:
   L_total = L_instance + lambda_prior * L_prior (lambda_prior = 1.0)
   Language Drift(클래스 지식 망각) 및 과적합(Overfitting)을 원천 차단하여
   Identity(CLIP-I)와 Text Alignment(CLIP-T)를 동시 극대화.

A100 40GB 최적화:
- T5-XXL + CLIP-L/G 텍스트 임베딩 및 VAE Latent 전량 사전 캐싱 (Pre-caching)
- SD3Transformer2DModel Attention LoRA (Rank 64, Alpha 64)
- 1,000 Steps 초고속 학습 (~60-80초/서브젝트)
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
# 1. Dataset & Pre-caching for Instance & Prior
# ==============================================================================

class DreamBoothConceptDataset(Dataset):
    """인스턴스 이미지와 클래스 사전(Prior) 이미지를 각각 로드하는 데이터셋"""

    def __init__(
        self,
        concept_dir: str,
        prior_dir: str,
        class_name: str,
        instance_token: str = "sks",
        image_size: int = 512,
    ):
        self.concept_dir = concept_dir
        self.prior_dir = prior_dir
        self.class_name = class_name
        self.instance_token = instance_token
        self.image_size = image_size

        # 1. Instance Samples 스캔
        self.instance_samples = []
        meta_path = os.path.join(concept_dir, "metadata.jsonl")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        img_path = os.path.join(concept_dir, data["file_name"])
                        if os.path.exists(img_path):
                            self.instance_samples.append((img_path, data.get("text", f"a photo of {instance_token} {class_name}")))

        if not self.instance_samples:
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
                self.instance_samples.append((img_p, caption))

        # _nobg 배경 제거 이미지에 가중치 부여 (배경 분리 강화)
        weighted_instance = []
        for img_p, caption in self.instance_samples:
            weighted_instance.append((img_p, caption))
            if "nobg" in os.path.basename(img_p):
                weighted_instance.append((img_p, caption))
        self.instance_samples = weighted_instance

        # 2. Class Prior Samples 스캔
        self.prior_samples = []
        if os.path.exists(prior_dir):
            prior_imgs = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                prior_imgs.extend(glob.glob(os.path.join(prior_dir, ext)))
            for img_p in sorted(prior_imgs):
                txt_p = os.path.splitext(img_p)[0] + ".txt"
                if os.path.exists(txt_p):
                    with open(txt_p, "r", encoding="utf-8") as tf:
                        caption = tf.read().strip()
                else:
                    caption = f"a photo of a {class_name}"
                self.prior_samples.append((img_p, caption))

        print(f"  [Dataset] 인스턴스 샘플: {len(self.instance_samples)}개, Class Prior 샘플: {len(self.prior_samples)}개")


def load_image_tensor(img_path: str, image_size: int = 512) -> Image.Image:
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return image


@torch.no_grad()
def precompute_dataset_cache(
    pipe: StableDiffusion3Pipeline,
    samples: List[Tuple[str, str]],
    desc: str,
    device: torch.device,
    dtype: torch.dtype,
    enable_t5: bool = True,
    image_size: int = 512,
) -> List[Dict[str, torch.Tensor]]:
    """VAE Latents 및 Text Embeddings를 사전 캐싱하여 메모리 및 속도 극대화"""
    cached = []
    print(f"⚡ {desc} 캐싱 중 (총 {len(samples)}개 샘플)...")

    for img_path, caption in tqdm(samples, desc=desc, leave=False):
        image = load_image_tensor(img_path, image_size)

        # 1. VAE Latent
        image_tensor = pipe.image_processor.preprocess(image).to(device=device, dtype=pipe.vae.dtype)
        posterior = pipe.vae.encode(image_tensor).latent_dist
        raw_latent = posterior.sample()
        shift_factor = pipe.vae.config.shift_factor
        scaling_factor = pipe.vae.config.scaling_factor
        latent = (raw_latent - shift_factor) * scaling_factor
        latent = latent.to(dtype=dtype)

        # 2. Text Embeddings
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=caption,
            prompt_2=caption,
            prompt_3=caption if enable_t5 else None,
            device=device,
            do_classifier_free_guidance=False
        )

        cached.append({
            "latent": latent.squeeze(0).cpu(),
            "prompt_embeds": prompt_embeds.squeeze(0).cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.squeeze(0).cpu(),
            "caption": caption,
        })

    return cached


# ==============================================================================
# 2. DreamBooth-LoRA Training Function
# ==============================================================================

def train_dreambooth_concept(
    concept: str,
    dataset_dir: str = "./augmentation",
    prior_base_dir: str = "./data/class_priors",
    output_base_dir: str = "./checkpoints/exp08_dreambooth_lora",
    instance_token: str = "sks",
    r: int = 64,
    lora_alpha: int = 64,
    lr: float = 5e-5,
    max_train_steps: int = 1000,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 2,
    prior_loss_weight: float = 1.0,
    weighting_scheme: str = "flow_shift",
    enable_t5: bool = True,
    seed: int = 42,
):
    if concept not in CLASS_PROMPT:
        print(f"[오류] 알 수 없는 서브젝트명: {concept}")
        return None

    class_name = CLASS_PROMPT[concept]
    concept_dir = os.path.join(dataset_dir, concept)
    prior_dir = os.path.join(prior_base_dir, concept)
    output_dir = os.path.join(output_base_dir, f"lora_{concept}")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"🌟 [{concept}] SD3.5 True DreamBooth-LoRA 학습 시작")
    print(f"  - Class: '{class_name}', Identifier: '{instance_token}'")
    print(f"  - Prior Loss Weight (lambda): {prior_loss_weight}")
    print(f"  - Rank: {r}, Alpha: {lora_alpha}, LR: {lr}, Max Steps: {max_train_steps}")
    print(f"  - T5 Text Encoder: {'활성화 (T5-XXL 4.7B)' if enable_t5 else '비활성화'}")
    print(f"  - 저장 경로: {output_dir}")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Pipeline 로드로 Latents & Embeddings 사전 캐싱
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    hf_token = os.getenv("HF_TOKEN")

    print(f"📦 VAE 및 텍스트 인코더 로드 중...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=hf_token
    ).to(device)

    dataset = DreamBoothConceptDataset(concept_dir, prior_dir, class_name, instance_token)
    instance_cached = precompute_dataset_cache(pipeline, dataset.instance_samples, f"Instance [{concept}]", device, dtype, enable_t5=enable_t5)
    prior_cached = precompute_dataset_cache(pipeline, dataset.prior_samples, f"Prior [{concept}]", device, dtype, enable_t5=enable_t5)

    if not prior_cached:
        print(f"⚠️ 경고: [{concept}] Prior 이미지가 없습니다! generate_class_priors.py를 먼저 실행하세요.")
        return None

    # 메모리 정리
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

    transformer.enable_gradient_checkpointing()

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

    # 3. DreamBooth Dual Loss 학습 루프
    transformer.train()
    progress_bar = tqdm(range(max_train_steps), desc=f"DreamBooth-LoRA [{concept}]")
    step = 0
    num_inst = len(instance_cached)
    num_prior = len(prior_cached)

    def sample_flow_timesteps(bs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if weighting_scheme == "logit_normal":
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=bs,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            ).to(device=device, dtype=dtype)
            sigmas = u
        else:
            u = torch.rand((bs,), device=device, dtype=dtype)
            shift = 3.0
            sigmas = shift * u / (1.0 + (shift - 1.0) * u)
        timesteps = sigmas * 1000.0
        return sigmas, timesteps

    start_train_time = time.time()

    while step < max_train_steps:
        # A) Instance Batch 샘플링
        inst_indices = torch.randint(0, num_inst, (batch_size,))
        inst_latents = torch.stack([instance_cached[i]["latent"] for i in inst_indices]).to(device=device, dtype=dtype)
        inst_prompt_embeds = torch.stack([instance_cached[i]["prompt_embeds"] for i in inst_indices]).to(device=device, dtype=dtype)
        inst_pooled_embeds = torch.stack([instance_cached[i]["pooled_prompt_embeds"] for i in inst_indices]).to(device=device, dtype=dtype)

        # Instance Flow Matching
        inst_sigmas, inst_timesteps = sample_flow_timesteps(batch_size)
        inst_noise = torch.randn_like(inst_latents)
        inst_sigmas_exp = inst_sigmas.view(-1, 1, 1, 1)
        inst_noisy_latents = (1.0 - inst_sigmas_exp) * inst_latents + inst_sigmas_exp * inst_noise
        inst_target = inst_noise - inst_latents

        inst_pred = transformer(
            hidden_states=inst_noisy_latents,
            timestep=inst_timesteps,
            encoder_hidden_states=inst_prompt_embeds,
            pooled_projections=inst_pooled_embeds,
            return_dict=False,
        )[0]

        loss_inst = F.mse_loss(inst_pred.float(), inst_target.float(), reduction="mean")

        # B) Class Prior Batch 샘플링
        prior_indices = torch.randint(0, num_prior, (batch_size,))
        prior_latents = torch.stack([prior_cached[i]["latent"] for i in prior_indices]).to(device=device, dtype=dtype)
        prior_prompt_embeds = torch.stack([prior_cached[i]["prompt_embeds"] for i in prior_indices]).to(device=device, dtype=dtype)
        prior_pooled_embeds = torch.stack([prior_cached[i]["pooled_prompt_embeds"] for i in prior_indices]).to(device=device, dtype=dtype)

        # Prior Flow Matching
        prior_sigmas, prior_timesteps = sample_flow_timesteps(batch_size)
        prior_noise = torch.randn_like(prior_latents)
        prior_sigmas_exp = prior_sigmas.view(-1, 1, 1, 1)
        prior_noisy_latents = (1.0 - prior_sigmas_exp) * prior_latents + prior_sigmas_exp * prior_noise
        prior_target = prior_noise - prior_latents

        prior_pred = transformer(
            hidden_states=prior_noisy_latents,
            timestep=prior_timesteps,
            encoder_hidden_states=prior_prompt_embeds,
            pooled_projections=prior_pooled_embeds,
            return_dict=False,
        )[0]

        loss_prior = F.mse_loss(prior_pred.float(), prior_target.float(), reduction="mean")

        # C) Total DreamBooth Loss: L_inst + lambda_prior * L_prior
        total_loss = loss_inst + prior_loss_weight * loss_prior
        total_loss = total_loss / gradient_accumulation_steps
        total_loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == max_train_steps:
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        step += 1
        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss_inst": f"{loss_inst.item():.4f}",
            "loss_prior": f"{loss_prior.item():.4f}",
            "total": f"{(total_loss.item() * gradient_accumulation_steps):.4f}",
            "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
        })

    elapsed = time.time() - start_train_time
    print(f"\n✓ [{concept}] 학습 완료 (소요 시간: {elapsed:.1f}초, {elapsed/60:.1f}분)")

    # 4. LoRA 가중치 저장
    print(f"💾 DreamBooth-LoRA 가중치 저장 중: {output_dir}")
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

    # 메타데이터 기록
    config_info = {
        "concept": concept,
        "class_name": class_name,
        "instance_token": instance_token,
        "method": "SD3.5 True DreamBooth-LoRA (Prior Preservation Loss)",
        "prior_loss_weight": prior_loss_weight,
        "rank": r,
        "alpha": lora_alpha,
        "lr": lr,
        "max_train_steps": max_train_steps,
        "enable_t5": enable_t5,
        "num_instance_samples": num_inst,
        "num_prior_samples": num_prior,
        "train_time_sec": elapsed,
    }
    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="SD3.5 True DreamBooth-LoRA Training")
    parser.add_argument("--concept", type=str, default="all", help="서브젝트명 또는 'all'")
    parser.add_argument("--dataset_dir", type=str, default="./augmentation")
    parser.add_argument("--prior_dir", type=str, default="./data/class_priors")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/exp08_dreambooth_lora")
    parser.add_argument("--instance_token", type=str, default="sks")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--prior_weight", type=float, default=0.3)
    parser.add_argument("--enable_t5", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    concepts_to_train = list(CLASS_PROMPT.keys()) if args.concept == "all" else [args.concept]

    total_start = time.time()
    for c in concepts_to_train:
        train_dreambooth_concept(
            concept=c,
            dataset_dir=args.dataset_dir,
            prior_base_dir=args.prior_dir,
            output_base_dir=args.output_dir,
            instance_token=args.instance_token,
            r=args.rank,
            lora_alpha=args.alpha,
            lr=args.lr,
            max_train_steps=args.steps,
            prior_loss_weight=args.prior_weight,
            enable_t5=args.enable_t5,
            seed=args.seed,
        )

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 75)
    print(f"🎉 모든 서브젝트 DreamBooth-LoRA 학습 완료! (총 소요 시간: {total_elapsed:.1f}초, {total_elapsed/60:.1f}분)")
    print("=" * 75)


if __name__ == "__main__":
    main()
