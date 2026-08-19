"""
Project-3 Automated Snapshot & Backup Manager
---------------------------------------------
실험(Iteration)별로 체크포인트, 생성 이미지, 평가 보고서를 덮어쓰지 않고
고유한 버전/타임스탬프 디렉토리로 안전하게 백업 및 동기화합니다.

사용법:
    1) 구글 드라이브로 스냅샷 백업 (타임스탬프 자동 생성):
       python backup_manager.py --target drive --tag "exp03_lora_base"

    2) 로컬 다운로드용 zip 파일 생성:
       python backup_manager.py --target zip --tag "exp03_lora_base"
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys


def get_timestamp_tag():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_to_drive(drive_base_dir: str = "/content/drive/MyDrive/project-3-snapshots", tag: str = ""):
    ts = get_timestamp_tag()
    snapshot_name = f"snapshot_{ts}_{tag}" if tag else f"snapshot_{ts}"
    dest_dir = os.path.join(drive_base_dir, snapshot_name)

    if not os.path.exists("/content/drive/MyDrive"):
        print("⚠️ Google Drive가 마운트되어 있지 않습니다. Colab에서 아래 코드로 먼저 마운트해 주세요:")
        print("from google.colab import drive\ndrive.mount('/content/drive')")
        return False

    os.makedirs(dest_dir, exist_ok=True)
    print(f"📦 Google Drive 스냅샷 백업 시작 -> {dest_dir}")

    targets = ["checkpoints", "experiments", "docs", "results"]
    for t in targets:
        if os.path.exists(t):
            dst = os.path.join(dest_dir, t)
            print(f"  - [{t}] 복사 중...")
            shutil.copytree(t, dst, dirs_exist_ok=True)

    print(f"✓ 스냅샷 백업 완료! (덮어쓰지 않고 '{snapshot_name}' 폴더에 독립 보관되었습니다.)")
    return True


def backup_to_zip(output_dir: str = "./backups", tag: str = ""):
    ts = get_timestamp_tag()
    snapshot_name = f"backup_{ts}_{tag}" if tag else f"backup_{ts}"
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f"{snapshot_name}.zip")

    print(f"📦 Zip 압축 아카이브 생성 중 -> {zip_path}")
    cmd = [
        "zip", "-r", zip_path,
        "checkpoints/", "experiments/", "docs/", "results/"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ 압축 파일 생성 완료: {zip_path}")
    print(f"💡 Colab에서 다운로드하려면: files.download('{zip_path}')")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Experiment Snapshot & Backup Manager")
    parser.add_argument("--target", type=str, default="zip", choices=["drive", "zip", "all"], help="백업 대상 (drive, zip, all)")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/project-3-snapshots", help="Google Drive 저장 경로")
    parser.add_argument("--tag", type=str, default="", help="실험 구분용 태그명 (예: exp03_lora_steps200)")

    args = parser.parse_args()

    print("=" * 60)
    print("      🔄 Project-3 버전 보존 백업 매니저")
    print("=" * 60)

    if args.target in ("drive", "all"):
        backup_to_drive(args.drive_dir, args.tag)
    if args.target in ("zip", "all"):
        backup_to_zip("./backups", args.tag)


if __name__ == "__main__":
    main()
