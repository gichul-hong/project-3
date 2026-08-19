"""
Class Prior Image Generator for DreamBooth Training
---------------------------------------------------
Generates generic class regularization images (e.g. "a photo of a person", "a photo of a cat")
using the frozen base SD3.5-medium model to prevent language drift and overfitting.
"""

import argparse
import os
import sys
import time
from typing import Dict

from dotenv import load_dotenv
from PIL import Image
import torch
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from tqdm.auto import tqdm

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

CLASS_PROMPT_TEMPLATES = {
    "actionfigure_2": ["a photo of an action figure", "an action figure on a plain background", "a high quality action figure"],
    "decoritems_woodenpot": ["a photo of a wooden pot", "a wooden flower pot on a table", "a handcrafted wooden pot"],
    "furniture_sofa2": ["a photo of a sofa", "a comfortable living room sofa", "a modern couch in a room"],
    "instrument_music2": ["a photo of a guitar", "an acoustic guitar standing", "a professional musical guitar"],
    "luggage_backpack1": ["a photo of a backpack", "a travel backpack on a plain background", "a school backpack"],
    "person_3": ["a photo of a person", "a portrait of an adult person", "a person outdoors smiling"],
    "pet_cat5": ["a photo of a cat", "a cute domestic cat sitting", "a cat looking at the camera"],
    "scene_waterfall": ["a photo of a waterfall", "a natural waterfall in the mountains", "water cascading down rocks"],
    "transport_tank": ["a photo of a military tank", "an armored combat tank in an open field", "a green military tank"],
    "wearable_jacket1": ["a photo of a jacket", "a warm winter jacket hanging", "a stylish modern jacket"],
}


def generate_class_priors(
    output_dir: str = "./data/class_priors",
    num_images_per_class: int = 50,
    batch_size: int = 4,
    num_inference_steps: int = 24,
    guidance_scale: float = 4.5,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    model_id = "stabilityai/stable-diffusion-3.5-medium"
    hf_token = os.getenv("HF_TOKEN")

    print("=" * 70)
    print("🎨 DreamBooth Class Prior 정규화 이미지 생성 파이프라인")
    print(f"  - 클래스당 생성 수: {num_images_per_class}장 (총 {len(CLASS_PROMPT) * num_images_per_class}장)")
    print(f"  - 추론 스텝: {num_inference_steps}, CFG: {guidance_scale}, 배치 크기: {batch_size}")
    print(f"  - 저장 경로: {output_dir}")
    print("=" * 70)

    print("📦 SD3.5-Medium 로딩 중...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=hf_token
    ).to(device)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)

    total_start = time.time()

    for concept, class_name in CLASS_PROMPT.items():
        concept_prior_dir = os.path.join(output_dir, concept)
        os.makedirs(concept_prior_dir, exist_ok=True)

        # 이미 충분한 이미지가 있으면 스킵
        existing = [f for f in os.listdir(concept_prior_dir) if f.endswith(('.png', '.jpg'))]
        if len(existing) >= num_images_per_class:
            print(f"✓ [{concept}] 기존 {len(existing)}개 Class Prior 이미지 확인 -> 스킵")
            continue

        templates = CLASS_PROMPT_TEMPLATES.get(concept, [f"a photo of a {class_name}"])
        print(f"\n▶ [{concept} ('{class_name}')] Prior 이미지 {num_images_per_class}장 생성 시작...")

        generated_count = len(existing)
        pbar = tqdm(total=num_images_per_class - generated_count, desc=f"Prior [{concept}]")

        while generated_count < num_images_per_class:
            cur_bs = min(batch_size, num_images_per_class - generated_count)
            # 템플릿 프롬프트 순환 배정
            batch_prompts = [templates[(generated_count + i) % len(templates)] for i in range(cur_bs)]
            
            gen_seed = seed + generated_count * 100
            generator = [torch.Generator(device=device).manual_seed(gen_seed + i) for i in range(cur_bs)]

            with torch.inference_mode():
                outputs = pipeline(
                    prompt=batch_prompts,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=512,
                    width=512,
                ).images

            for i, img in enumerate(outputs):
                img_idx = generated_count + i
                img_path = os.path.join(concept_prior_dir, f"prior_{img_idx:03d}.png")
                txt_path = os.path.join(concept_prior_dir, f"prior_{img_idx:03d}.txt")
                img.save(img_path)
                with open(txt_path, "w", encoding="utf-8") as tf:
                    tf.write(batch_prompts[i])

            generated_count += cur_bs
            pbar.update(cur_bs)

        pbar.close()
        print(f"✓ [{concept}] {num_images_per_class}개 Class Prior 이미지 생성 완료 -> {concept_prior_dir}")

    total_elapsed = time.time() - total_start
    print(f"\n🎉 전체 Class Prior 생성 완료! (총 소요 시간: {total_elapsed:.1f}초, {total_elapsed/60:.1f}분)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/class_priors")
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_class_priors(
        output_dir=args.output_dir,
        num_images_per_class=args.num_images,
        batch_size=args.batch_size,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
    )
