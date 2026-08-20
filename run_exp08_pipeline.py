"""
Master Pipeline for Exp-08: True DreamBooth-LoRA & Extended Multi-Metric Evaluation
----------------------------------------------------------------------------------
1. Phase 1: Class Prior Regularization Images Generation (SD3.5 Base, 40 imgs/class)
2. Phase 2: SD3.5 True DreamBooth-LoRA Training (Dual Flow Loss, Rank 64, 1000 steps)
   - Real-time Google Drive sync & per-concept auto-resume
   - Per-concept git push upon completion
3. Phase 3: Controlled ODE Hybrid Inference (Exp-08 weights + Adaptive eta + Multi-ref avg + Heun 50 steps)
   - Real-time Drive sync & per-concept git push
4. Phase 4: Extended Multi-Metric Evaluation (CLIP-T, CLIP-I, DINOv2-I, 4-Axis Taxonomy)
5. Phase 5: Experiment History & Viewer Dashboard Update
"""

import argparse
import glob
import json
import os
import shutil
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

DRIVE_BACKUP_BASE = "/content/drive/MyDrive/project-3-backup"


def sync_to_drive(src_path: str, rel_dest: str):
    """구글 드라이브 백업 디렉토리에 파일/폴더를 안전하게 실시간 동기화"""
    if not os.path.exists("/content/drive/MyDrive"):
        return
    dest_path = os.path.join(DRIVE_BACKUP_BASE, rel_dest)
    try:
        if os.path.isdir(src_path):
            os.makedirs(dest_path, exist_ok=True)
            for item in os.listdir(src_path):
                s = os.path.join(src_path, item)
                d = os.path.join(dest_path, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
        elif os.path.isfile(src_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)
        print(f"  ☁️ [Drive Sync] -> {dest_path} 동기화 완료")
    except Exception as e:
        print(f"  ⚠️ [Drive Sync 경고] {e}")


def sync_from_drive(rel_src: str, local_dest: str):
    """구글 드라이브에 보관된 파일/폴더를 로컬로 복원"""
    drive_src = os.path.join(DRIVE_BACKUP_BASE, rel_src)
    if not os.path.exists(drive_src):
        return False
    try:
        if os.path.isdir(drive_src):
            os.makedirs(local_dest, exist_ok=True)
            for item in os.listdir(drive_src):
                s = os.path.join(drive_src, item)
                d = os.path.join(local_dest, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
        elif os.path.isfile(drive_src):
            os.makedirs(os.path.dirname(local_dest), exist_ok=True)
            shutil.copy2(drive_src, local_dest)
        print(f"  📥 [Drive Restore] {drive_src} -> {local_dest} 복원 완료")
        return True
    except Exception as e:
        print(f"  ⚠️ [Drive Restore 경고] {e}")
        return False


def auto_git_push(msg: str):
    try:
        print(f"\n🔄 [Git Auto-Sync] '{msg}' 커밋 및 푸시 중...")
        subprocess.run(["git", "add", "."], check=False)
        subprocess.run(["git", "commit", "-m", f"chore: {msg}"], check=False)
        res = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
        if res.returncode == 0:
            print("✓ Git 푸시 완료!")
        else:
            print(f"⚠️ Git 푸시 출력: {res.stderr.strip()}")
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


def is_valid_lora_checkpoint(ckpt_dir: str) -> bool:
    """LoRA 체크포인트 유효성 검사 (safetensors 파일 존재 및 크기 1MB 이상)"""
    if not os.path.exists(ckpt_dir):
        return False
    safetensors = glob.glob(os.path.join(ckpt_dir, "*.safetensors"))
    for sf in safetensors:
        if os.path.getsize(sf) > 1024 * 1024:  # > 1MB
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Master Pipeline for Exp-08")
    parser.add_argument("--skip_priors", action="store_true", help="Class prior 생성 건너뛰기")
    parser.add_argument("--skip_train", action="store_true", help="DreamBooth 학습 건너뛰기")
    parser.add_argument("--steps_train", type=int, default=1000)
    parser.add_argument("--steps_gen", type=int, default=50)
    args = parser.parse_args()

    total_start = time.time()

    # 0. 초기 체크포인트 복원 시도 (구글 드라이브 -> 로컬)
    print("📦 구글 드라이브 백업 상태 점검 중...")
    sync_from_drive("checkpoints/exp08_dreambooth_lora", "./checkpoints/exp08_dreambooth_lora")
    sync_from_drive("data/class_priors", "./data/class_priors")

    # Phase 1: Class Prior Images Generation
    if not args.skip_priors:
        priors_needed = False
        for c in CONCEPTS:
            prior_concept_dir = f"./data/class_priors/{c}"
            imgs = glob.glob(os.path.join(prior_concept_dir, "*.png"))
            if len(imgs) < 40:
                priors_needed = True
                break

        if priors_needed:
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
            sync_to_drive("./data/class_priors", "data/class_priors")
            auto_git_push("complete Phase 1 Class Priors generation")
        else:
            print("✓ [Phase 1] 10개 클래스 전체 Class Prior 이미지(40장/클래스) 이미 존재 -> 스킵")

    # Phase 2: True DreamBooth-LoRA Training (10 concepts) with Per-concept Resume & Sync
    if not args.skip_train:
        for c in CONCEPTS:
            local_ckpt = f"./checkpoints/exp08_dreambooth_lora/lora_{c}"
            drive_ckpt = f"checkpoints/exp08_dreambooth_lora/lora_{c}"

            # 1) 로컬 또는 드라이브에 이미 유효한 체크포인트가 있는지 확인
            if not is_valid_lora_checkpoint(local_ckpt):
                sync_from_drive(drive_ckpt, local_ckpt)

            if is_valid_lora_checkpoint(local_ckpt):
                print(f"✓ [{c}] DreamBooth-LoRA 체크포인트 이미 존재 -> 학습 스킵 ({local_ckpt})")
                continue

            # 2) 학습 실행
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
                    "--prior_weight", "0.3",
                    "--enable_t5",
                ],
                f"Phase 2: True DreamBooth-LoRA 학습 [{c}] (Prior Loss lambda=0.3, Rank 64, {args.steps_train} Steps)",
            )

            # 3) 즉시 구글 드라이브 동기화 및 깃 푸시
            sync_to_drive(local_ckpt, drive_ckpt)
            auto_git_push(f"train exp08 True DreamBooth-LoRA [{c}]")

        auto_git_push("complete Phase 2 True DreamBooth-LoRA 10 concepts training")

    # Phase 3-A: Exp-08 Controlled ODE Inference & CLIP Evaluation (10 concepts)
    exp08_out_dir = "./experiments/08_dreambooth_prior_loss"
    os.makedirs(exp08_out_dir, exist_ok=True)

    for c in CONCEPTS:
        gen_imgs = glob.glob(os.path.join(exp08_out_dir, c, "*.png"))
        if len(gen_imgs) >= 10:
            print(f"✓ [{c}] Exp-08 생성 이미지 10장 이미 존재 -> 추론 스킵 ({len(gen_imgs)}장)")
            continue

        run_cmd(
            [
                sys.executable,
                "generate_hybrid.py",
                "--concept", c,
                "--checkpoints_dir", "./checkpoints/exp08_dreambooth_lora",
                "--output", exp08_out_dir,
                "--dataset", "./dataset",
                "--aug_dir", "./augmentation",
                "--prompts", "./prompt",
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
            f"Phase 3-A: Exp-08 DreamBooth + Null-Text Heun 50-Step 추론 [{c}]",
        )

        sync_to_drive(os.path.join(exp08_out_dir, c), f"experiments/08_dreambooth_prior_loss/{c}")
        auto_git_push(f"generate exp08 images [{c}]")

    sync_to_drive(exp08_out_dir, "experiments/08_dreambooth_prior_loss")
    auto_git_push("complete Phase 3-A Exp-08 generation & evaluation")

    # Phase 3-B: Exp-09 Subject-Aware Dynamic Routing Hybrid Inference (10 concepts)
    exp09_out_dir = "./experiments/09_subject_adaptive_routing"
    os.makedirs(exp09_out_dir, exist_ok=True)

    for c in CONCEPTS:
        gen_imgs = glob.glob(os.path.join(exp09_out_dir, c, "*.png"))
        if len(gen_imgs) >= 10:
            print(f"✓ [{c}] Exp-09 생성 이미지 10장 이미 존재 -> 추론 스킵 ({len(gen_imgs)}장)")
            continue

        run_cmd(
            [
                sys.executable,
                "generate_hybrid.py",
                "--concept", c,
                "--checkpoints_dir", "./checkpoints/exp08_dreambooth_lora",
                "--output", exp09_out_dir,
                "--dataset", "./dataset",
                "--aug_dir", "./augmentation",
                "--prompts", "./prompt",
                "--ref_mode", "avg",
                "--eta_schedule", "adaptive",
                "--scheduler", "heun",
                "--subject_routing",
                "--steps", str(args.steps_gen),
                "--enable_t5",
                "--custom_neg",
                "--seed", "42",
            ],
            f"Phase 3-B: Exp-09 Subject-Aware Dynamic Routing Heun 50-Step 추론 [{c}]",
        )

        sync_to_drive(os.path.join(exp09_out_dir, c), f"experiments/09_subject_adaptive_routing/{c}")
        auto_git_push(f"generate exp09 images [{c}]")

    sync_to_drive(exp09_out_dir, "experiments/09_subject_adaptive_routing")
    auto_git_push("complete Phase 3-B Exp-09 generation & evaluation")

    # Phase 4: Extended Multi-Metric Evaluation across all experiments
    run_cmd(
        [
            sys.executable,
            "evaluate_extended.py",
            "--exp_dir", "all",
            "--data_dir", "./dataset",
        ],
        "Phase 4: 전체 실험 다차원 정밀 평가 (DINO-v2 + 4대 Taxonomy + Diversity)",
    )

    # 보고서 갱신
    print("\n📝 전체 실험 보고서 (EVALUATION_REPORT.md / README.md) 갱신 중...")
    subprocess.run([sys.executable, "update_all_reports.py"])
    sync_to_drive("./docs", "docs")
    sync_to_drive("./results", "results")
    auto_git_push("complete Phase 4 extended multi-metric evaluation & reports")

    # Phase 5: Experiment Viewer Dashboard 갱신
    print("\n📊 웹 대시보드 (experiment_viewer.html) 갱신 중...")
    subprocess.run([sys.executable, "generate_experiment_viewer.py"])
    sync_to_drive("./experiment_viewer.html", "experiment_viewer.html")
    auto_git_push("complete Phase 5 dashboard & report update")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"🎉 Exp-08 및 종합 평가 파이프라인 100% 완료! (총 소요 시간: {total_elapsed:.1f}초, {total_elapsed/60:.1f}분)")
    print("=" * 80)


if __name__ == "__main__":
    main()
