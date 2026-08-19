"""
SD3.5-Medium LoRA Generation & CLIP Evaluation Pipeline
-------------------------------------------------------
학습된 LoRA 체크포인트를 로드하여 10개 테스트 프롬프트 기반 이미지를 생성하고
CLIP-T / CLIP-I 평가를 수행하여 리포트를 생성합니다.

사용법:
    1) 샘플 서브젝트 생성 및 평가:
       python generate_lora.py --concept actionfigure_2

    2) 전체 10개 서브젝트 일괄 생성 및 평가:
       python generate_lora.py --concept all --output ./experiments/03_lora_augmented
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image
import torch

from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
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


def load_sd3_lora_pipeline(checkpoint_dir: str, device_type: str = "cuda"):
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
        print(f"🔗 LoRA 가중치 로드 중: {checkpoint_dir}")
        try:
            # PEFT 방식으로 transformer에 LoRA 주입
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

    pipeline.set_progress_bar_config(desc="LoRA Generating")
    return pipeline


def generate_for_concept(
    pipeline: StableDiffusion3Pipeline,
    concept: str,
    prompts_dir: str = "./prompt",
    output_dir: str = "./generated",
    instance_token: str = "sks",
    use_token: bool = True,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
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
    token_word = f"{instance_token} {class_word}" if use_token else class_word

    print(f"\n▶ [{concept}] LoRA 기반 생성 시작 (단어: '{token_word}')")

    for idx, raw_p in enumerate(prompts_raw):
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
    parser = argparse.ArgumentParser(description="SD3.5 LoRA Generation & Evaluation")
    parser.add_argument("--concept", type=str, default="actionfigure_2", help="서브젝트명 ('all' 또는 특정 서브젝트)")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="체크포인트 상위 디렉토리")
    parser.add_argument("--dataset", type=str, default="./dataset", help="평가용 원본 레퍼런스 데이터셋 경로")
    parser.add_argument("--prompts", type=str, default="./prompt", help="프롬프트 폴더 경로")
    parser.add_argument("--output", type=str, default="./experiments/03_lora_augmented", help="생성 결과 및 보고서 저장 폴더")
    parser.add_argument("--instance_token", type=str, default="sks", help="인스턴스 토큰")
    parser.add_argument("--no_token", action="store_true", help="프롬프트 생성 시 instance token 제외 여부")
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
        pipeline = load_sd3_lora_pipeline(checkpoint_dir=ckpt_dir, device_type=device)

        generate_for_concept(
            pipeline=pipeline,
            concept=concept,
            prompts_dir=args.prompts,
            output_dir=args.output,
            instance_token=args.instance_token,
            use_token=not args.no_token,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            seed=args.seed
        )

        # 메모리 정리
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
        avg_t2i = round(sum(v["t2i"] for v in eval_results.values()) / len(eval_results), 4)
        avg_i2i = round(sum(v["i2i"] for v in eval_results.values()) / len(eval_results), 4)

        summary_data = {
            "method": "SD3.5 LoRA Fine-Tuning",
            "instance_token": args.instance_token if not args.no_token else None,
            "hyperparameters": {
                "steps": args.steps,
                "cfg": args.cfg,
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

        # JSON & Markdown 동시 저장
        out_json_path = os.path.join(args.output, "eval_summary.json")
        out_md_path = os.path.join(args.output, "EVALUATION_REPORT.md")

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        report_content = (
            "# 📊 Subject-driven SD3.5 LoRA Fine-Tuning Evaluation Report\n\n"
            f"- **실행 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- **소요 시간**: {elapsed:.1f}초 ({elapsed/60:.1f}분)\n"
            f"- **방법론**: `SD3.5 LoRA Fine-Tuning (Augmented Dataset)`\n"
            f"- **하이퍼파라미터**: Steps={args.steps}, CFG={args.cfg}, Token='{args.instance_token}', Seed={args.seed}\n\n"
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

    print("\n🎉 모든 LoRA 생성 및 평가 과정이 완료되었습니다!")


if __name__ == "__main__":
    main()
