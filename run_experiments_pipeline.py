"""
Subject-driven Customization Master Pipeline (Exp-05, Exp-06, Exp-07)
---------------------------------------------------------------------
3단 LoRA / DreamBooth + RF-Inversion iteration을 일괄 또는 단계별로 실행하고
CLIP-T / CLIP-I 평가 및 웹 대시보드(experiment_viewer.html)를 자동 갱신합니다.

실험 구성:
1. Exp-05: SD3.5 LoRA 고품질 (Rank 64, Alpha 64, T5-XXL 활성화, Steps 500)
2. Exp-06: Exp-05 LoRA + Controlled ODE (Adaptive eta schedule + Multi-reference Latent averaging)
3. Exp-07: Exp-06 + FlowMatch Heun 2nd-order solver (50 steps) + Concept별 맞춤 Negative Prompt

사용법:
    # 1) 전체 3단계 파이프라인 일괄 실행 (10개 서브젝트)
    python run_experiments_pipeline.py --mode all --steps_train 500

    # 2) 샘플 서브젝트 (actionfigure_2, pet_cat5) 빠른 검증
    python run_experiments_pipeline.py --mode sample --steps_train 300

    # 3) 특정 실험 단계만 개별 실행
    python run_experiments_pipeline.py --mode exp05 --concept all
    python run_experiments_pipeline.py --mode exp06 --concept all
    python run_experiments_pipeline.py --mode exp07 --concept all
"""

import argparse
import json
import os
import subprocess
import sys
import time

SAMPLE_CONCEPTS = ["actionfigure_2", "pet_cat5"]

ALL_CONCEPTS = [
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


def run_cmd(cmd, desc=""):
    print("\n" + "=" * 80)
    print(f"▶ [실행] {desc}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 80)
    start_t = time.time()
    res = subprocess.run(cmd, check=True)
    elapsed = time.time() - start_t
    print(f"✓ [{desc}] 완료 (소요 시간: {elapsed:.1f}초, {elapsed/60:.1f}분)")
    return elapsed


def update_dashboard():
    print("\n📊 웹 대시보드 (experiment_viewer.html) 갱신 중...")
    cmd = [sys.executable, "generate_experiment_viewer.py"]
    subprocess.run(cmd, check=True)
    print("✓ experiment_viewer.html 갱신 완료!")


def run_exp05(concepts_arg, steps_train=500, steps_gen=28, seed=42):
    print("\n" + "#" * 80)
    print("🚀 [Exp-05] SD3.5 High-Quality LoRA Fine-Tuning & Evaluation")
    print(f"   Concepts: {concepts_arg}, Train Steps: {steps_train}, Rank: 64, Alpha: 64, T5: Active")
    print("#" * 80)

    # 1. Train LoRA for each concept
    if concepts_arg == "all":
        target_list = ALL_CONCEPTS
    elif concepts_arg == "sample":
        target_list = SAMPLE_CONCEPTS
    else:
        target_list = [concepts_arg]

    for concept in target_list:
        cmd_train = [
            sys.executable, "train_lora_sd3.py",
            "--concept", concept,
            "--exp_name", "exp05_lora_hq",
            "--rank", "64",
            "--alpha", "64",
            "--steps", str(steps_train),
            "--enable_t5",
            "--seed", str(seed),
        ]
        run_cmd(cmd_train, f"Exp-05 Train LoRA [{concept}]")

    # 2. Generate & Evaluate in batch
    for concept in target_list:
        cmd_gen = [
            sys.executable, "generate_lora.py",
            "--concept", concept,
            "--exp_name", "exp05_lora_hq",
            "--output", "./experiments/05_lora_hq",
            "--steps", str(steps_gen),
            "--enable_t5",
            "--custom_neg",
            "--seed", str(seed),
        ]
        run_cmd(cmd_gen, f"Exp-05 Generate & Evaluate [{concept}]")


def run_exp06(concepts_arg, steps_gen=28, tau=0.7, eta=0.8, seed=42):
    print("\n" + "#" * 80)
    print("🚀 [Exp-06] LoRA + Controlled ODE Hybrid (Adaptive eta + Multi-reference avg)")
    print(f"   Concepts: {concepts_arg}, tau: {tau}, eta: {eta}, schedule: adaptive, ref: avg")
    print("#" * 80)

    if concepts_arg == "all":
        target_list = ALL_CONCEPTS
    elif concepts_arg == "sample":
        target_list = SAMPLE_CONCEPTS
    else:
        target_list = [concepts_arg]

    for concept in target_list:
        cmd_hybrid = [
            sys.executable, "generate_hybrid.py",
            "--concept", concept,
            "--checkpoints_dir", "./checkpoints/exp05_lora_hq",
            "--output", "./experiments/06_hybrid_adaptive",
            "--ref_mode", "avg",
            "--eta_schedule", "adaptive",
            "--scheduler", "euler",
            "--tau", str(tau),
            "--eta", str(eta),
            "--steps", str(steps_gen),
            "--enable_t5",
            "--custom_neg",
            "--seed", str(seed),
        ]
        run_cmd(cmd_hybrid, f"Exp-06 Hybrid Generate & Evaluate [{concept}]")


def run_exp07(concepts_arg, steps_gen=50, tau=0.7, eta=0.85, seed=42):
    print("\n" + "#" * 80)
    print("🚀 [Exp-07] LoRA + Heun 50-Step Solver + Custom Negative Prompt")
    print(f"   Concepts: {concepts_arg}, Steps: {steps_gen} (Heun), tau: {tau}, eta: {eta}")
    print("#" * 80)

    if concepts_arg == "all":
        target_list = ALL_CONCEPTS
    elif concepts_arg == "sample":
        target_list = SAMPLE_CONCEPTS
    else:
        target_list = [concepts_arg]

    for concept in target_list:
        cmd_heun = [
            sys.executable, "generate_hybrid.py",
            "--concept", concept,
            "--checkpoints_dir", "./checkpoints/exp05_lora_hq",
            "--output", "./experiments/07_heun_custom_neg",
            "--ref_mode", "avg",
            "--eta_schedule", "adaptive",
            "--scheduler", "heun",
            "--tau", str(tau),
            "--eta", str(eta),
            "--steps", str(steps_gen),
            "--enable_t5",
            "--custom_neg",
            "--seed", str(seed),
        ]
        run_cmd(cmd_heun, f"Exp-07 Heun 50-Step Generate & Evaluate [{concept}]")


def main():
    parser = argparse.ArgumentParser(description="SD3.5 Iteration Pipeline Master Runner")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "sample", "exp05", "exp06", "exp07", "viewer_only"], help="실행 모드")
    parser.add_argument("--concept", type=str, default="all", help="서브젝트명 ('all', 'sample', 또는 특정 서브젝트)")
    parser.add_argument("--steps_train", type=int, default=500, help="LoRA 학습 스텝 수 (기본 500)")
    parser.add_argument("--steps_gen", type=int, default=28, help="표준 인퍼런스 스텝 수 (기본 28)")
    parser.add_argument("--steps_heun", type=int, default=50, help="Heun 인퍼런스 스텝 수 (기본 50)")
    parser.add_argument("--seed", type=int, default=42, help="시드값")

    args = parser.parse_args()

    if args.mode == "viewer_only":
        update_dashboard()
        return

    concept_target = "sample" if args.mode == "sample" else args.concept

    total_start = time.time()

    if args.mode in ["all", "sample", "exp05"]:
        run_exp05(concept_target, steps_train=args.steps_train, steps_gen=args.steps_gen, seed=args.seed)

    if args.mode in ["all", "sample", "exp06"]:
        run_exp06(concept_target, steps_gen=args.steps_gen, seed=args.seed)

    if args.mode in ["all", "sample", "exp07"]:
        run_exp07(concept_target, steps_gen=args.steps_heun, seed=args.seed)

    update_dashboard()

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"🎉 모든 실험 및 대시보드 갱신 완료! (총 소요 시간: {total_elapsed:.1f}초, {total_elapsed/60:.1f}분)")
    print("=" * 80)


if __name__ == "__main__":
    main()
