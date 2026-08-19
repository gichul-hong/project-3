"""
Zero-Shot Baseline Image Generation Script (No Fine-tuning)
------------------------------------------------------------
파인튜닝(DreamBooth/LoRA) 및 Inversion 조작 없이, SD3.5-medium 원본 모델을 이용하여 
테스트 프롬프트 기반 Zero-Shot 생성을 수행하고 CLIP-T / CLIP-I Baseline 점수를 측정합니다.

사용법:
    1) 샘플 서브젝트 1개만 실행 (빠른 테스트):
       python generate_baseline.py --concept actionfigure_2

    2) 전체 10개 서브젝트 실행:
       python generate_baseline.py --concept all

    3) 생성 후 evaluation.py 자동 평가 수행
"""

import argparse
import os
import subprocess
import sys
from dotenv import load_dotenv
import torch
from PIL import Image

# .env 파일에서 HF_TOKEN 로드 (로컬 환경)
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


def load_sd3_pipeline(device_type="cuda"):
    from diffusers import StableDiffusion3Pipeline

    model_id = "stabilityai/stable-diffusion-3.5-medium"
    print(f"📦 모델 로딩 시작 ({model_id})...")

    # HuggingFace 토큰 확인
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("✓ HF_TOKEN 감지됨 (HuggingFace 게이팅 인증 사용)")

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

    if device_type == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to("cpu")

    pipeline.set_progress_bar_config(desc="Zero-Shot Generating")
    print("✓ SD3.5-medium 파이프라인 로드 완료!")
    return pipeline


def generate_baseline_for_concept(
    pipeline,
    concept: str,
    prompts_dir: str = "./prompt",
    output_dir: str = "./generated",
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    seed: int = 42
):
    if concept not in CLASS_PROMPT:
        print(f"[오류] 알 수 없는 서브젝트명입니다: {concept}")
        return

    class_word = CLASS_PROMPT[concept]
    prompt_file = os.path.join(prompts_dir, f"{concept}.txt")

    if not os.path.exists(prompt_file):
        print(f"[오류] 프롬프트 파일이 없습니다: {prompt_file}")
        return

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts_raw = [l.strip() for l in f.readlines() if l.strip()]

    concept_out_dir = os.path.join(output_dir, concept)
    os.makedirs(concept_out_dir, exist_ok=True)

    print(f"\n▶ [{concept}] Zero-Shot 생성 시작 (프롬프트 {len(prompts_raw)}개, class: '{class_word}')")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for idx, raw_p in enumerate(prompts_raw):
        # {} 를 class prompt 단어로 치환
        prompt_text = raw_p.replace("{}", class_word)
        out_path = os.path.join(concept_out_dir, f"{idx}.png")

        print(f"  [{idx}/9] 프롬프트: \"{prompt_text}\"")

        generator = torch.Generator(device=device).manual_seed(seed + idx)

        image = pipeline(
            prompt=prompt_text,
            negative_prompt="low quality, bad resolution, distorted",
            num_inference_steps=num_inference_steps,
            height=512,
            width=512,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        image.save(out_path)
        print(f"      -> 저장 완료: {out_path}")

    print(f"✓ [{concept}] 10장 이미지 생성 완료 -> {concept_out_dir}")


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
    except subprocess.CalledProcessError as e:
        print(f"[오류] Evaluation 실행 중 문제 발생:\n{e.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot SD3.5 Baseline Image Generation")
    parser.add_argument("--concept", type=str, default="actionfigure_2", help="서브젝트명 ('all' 또는 특정 서브젝트)")
    parser.add_argument("--dataset", type=str, default="./dataset", help="레퍼런스 데이터셋 경로")
    parser.add_argument("--prompts", type=str, default="./prompt", help="프롬프트 폴더 경로")
    parser.add_argument("--output", type=str, default="./generated", help="생성 이미지 저장 경로")
    parser.add_argument("--steps", type=int, default=28, help="인퍼런스 스텝 수 (기본 28)")
    parser.add_argument("--cfg", type=float, default=7.0, help="Guidance scale (기본 7.0)")
    parser.add_argument("--seed", type=int, default=42, help="시드값 (기본 42)")
    parser.add_argument("--no_eval", action="store_true", help="생성 후 evaluation 자동 평가 비활성화")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("   SD3.5 Zero-Shot Baseline 생성 파이프라인 (No Finetuning)")
    print("=" * 60)
    print(f"- Device: {device}")
    print(f"- 대상 서브젝트: {args.concept}")
    print(f"- Steps: {args.steps}, CFG: {args.cfg}")

    pipeline = load_sd3_pipeline(device_type=device)

    if args.concept == "all":
        target_concepts = list(CLASS_PROMPT.keys())
    else:
        target_concepts = [args.concept]

    for concept in target_concepts:
        generate_baseline_for_concept(
            pipeline=pipeline,
            concept=concept,
            prompts_dir=args.prompts,
            output_dir=args.output,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            seed=args.seed
        )
        if not args.no_eval:
            run_evaluation(
                concept=concept,
                dataset_dir=args.dataset,
                prompts_dir=args.prompts,
                generated_dir=args.output
            )

    print("\n" + "=" * 60)
    print("🎉 Zero-Shot Baseline 생성 및 평가 과정이 완료되었습니다!")
    print("=" * 60)


if __name__ == "__main__":
    main()
