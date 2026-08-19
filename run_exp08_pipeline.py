"""
Master Pipeline for Exp-08: True DreamBooth-LoRA & Extended Multi-Metric Evaluation
----------------------------------------------------------------------------------
1. Phase 1: Class Prior Regularization Images Generation (SD3.5 Base, 50 imgs/class)
2. Phase 2: SD3.5 True DreamBooth-LoRA Training (Dual Flow Loss, Rank 64, 1000 steps)
3. Phase 3: Controlled ODE Hybrid Inference (Exp-08 weights + Adaptive eta + Multi-ref avg + Heun 50 steps)
4. Phase 4: Extended Multi-Metric Evaluation (CLIP-T, CLIP-I, DINOv2-I, 4-Axis Taxonomy)
5. Phase 5: Experiment History & Viewer Dashboard Update
"""

import argparse
import json
import os
import subprocess
import sys
import time

CONCEPTS = [
    "actionfigure_2",
    "decoritems_woodenpot",
    "furniture_sofa2",
    "instrument_music2",
    "luggage_backpack1",
    "person_3",
    "pet_cat5",
    "scene_waterfall",
    "transport_tank",
    "wearable_jacket1",
]


def auto_git_push(msg: str):
    try:
        print(f"\n🔄 [Git Auto-Sync] '{msg}' 커밋 및 푸시 중...")
        subprocess.run(["git", "add", "."], check=False)
        subprocess.run(["git", "commit", "-m", f"chore: {msg}"], check=False)
        subprocess.run(["git", "push", "origin", "main"], check=False)
        print("✓ Git 푸시 완료!")
    except Exception as e:
        print(f"⚠️ Git 푸시 스킵: {e}")


def run_cmd(cmd: list, desc: str):
    print("\n" + "=" * 80)
    print(f"▶ [실행] {desc}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 80)
    start_t = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start_t
    if res.returncode != 0:
        print(f"❌ [오류 발생] {desc} 실패 (Exit code: {res.returncode})")
        sys.exit(res.returncode)
    print(f"✓ [{desc}] 완료 (소요 시간: {elapsed:.1f}초, {elapsed/60:.1f}분)\n")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_priors", action="store_true", help="Class prior 생성 건너뛰기")
    parser.add_argument("--skip_train", action="store_true", help="DreamBooth 학습 건너뛰기")
    parser.add_argument("--steps_train", type=int, default=1000)
    parser.add_argument("--steps_gen", type=int, default=50)
    args = parser.parse_args()

    total_start = time.time()

    # Phase 1: Class Prior Images Generation
    if not args.skip_priors:
        run_cmd(
            [
                sys.executable,
                "generate_class_priors.py",
                "--output_dir", "./data/class_priors",
                "--num_images", "40",
                "--batch_size", "4",
                "--steps", "24",
                "--cfg", "4.5",
                "--seed", "42",
            ],
            "Phase 1: Class Prior 정규화 이미지 고속 생성 (10개 클래스 x 40장)",
        )
        auto_git_push("complete Phase 1 Class Priors generation")

    # Phase 2: True DreamBooth-LoRA Training (10 concepts)
    if not args.skip_train:
        for c in CONCEPTS:
            run_cmd(
                [
                    sys.executable,
                    "train_dreambooth_sd3.py",
                    "--concept", c,
                    "--dataset_dir", "./augmentation",
                    "--prior_dir", "./data/class_priors",
                    "--output_dir", "./checkpoints/exp08_dreambooth_lora",
                    "--rank", "64",
                    "--alpha", "64",
                    "--lr", "5e-5",
                    "--steps", str(args.steps_train),
                    "--prior_weight", "1.0",
                    "--enable_t5",
                ],
                f"Phase 2: True DreamBooth-LoRA 학습 [{c}] (Prior Loss lambda=1.0, Rank 64, {args.steps_train} Steps)",
            )
        auto_git_push("complete Phase 2 True DreamBooth-LoRA 10 concepts training")

    # Phase 3: Exp-08 Controlled ODE Inference & CLIP Evaluation (10 concepts)
    for c in CONCEPTS:
        run_cmd(
            [
                sys.executable,
                "generate_hybrid.py",
                "--concept", c,
                "--checkpoints_dir", "./checkpoints/exp08_dreambooth_lora",
                "--output", "./experiments/08_dreambooth_prior_loss",
                "--ref_mode", "avg",
                "--eta_schedule", "adaptive",
                "--scheduler", "heun",
                "--tau", "0.7",
                "--eta", "0.85",
                "--steps", str(args.steps_gen),
                "--enable_t5",
                "--custom_neg",
                "--seed", "42",
            ],
            f"Phase 3: Exp-08 DreamBooth + Controlled ODE Heun 50-Step 추론 [{c}]",
        )
    auto_git_push("complete Phase 3 Exp-08 generation & evaluation")

    # Phase 4: Extended Multi-Metric Evaluation across all experiments
    run_cmd(
        [
            sys.executable,
            "evaluate_extended.py",
            "--exp_dir", "all",
            "--data_dir", "./data",
        ],
        "Phase 4: 전체 실험 다차원 정밀 평가 (DINO-v2 + 4대 Taxonomy + Diversity)",
    )
    auto_git_push("complete Phase 4 extended multi-metric evaluation")

    # Phase 5: Experiment Viewer Dashboard 갱신
    print("\n📊 웹 대시보드 (experiment_viewer.html) 갱신 중...")
    subprocess.run([sys.executable, "generate_experiment_viewer.py"])
    print("✓ experiment_viewer.html 갱신 완료!")
    auto_git_push("complete Phase 5 dashboard & report update")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"🎉 Exp-08 및 종합 평가 파이프라인 100% 완료! (총 소요 시간: {total_elapsed:.1f}초, {total_elapsed/60:.1f}분)")
    print("=" * 80)


if __name__ == "__main__":
    main()
