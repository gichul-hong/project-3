"""
CustomConcept101 Dataset Augmentation Script
----------------------------------------------
사용법:
  1) 전체 카테고리(10개) 증강:
     python augment_dataset.py --dataset ./dataset --output ./augmentation --concept all

  2) 특정 카테고리만 지정 증강 (예: actionfigure_2):
     python augment_dataset.py --dataset ./dataset --output ./augmentation --concept actionfigure_2

  3) 배경 제거(rembg) 활성화 (rembg 패키지 설치 시 자동 적용):
     pip install rembg
     python augment_dataset.py --dataset ./dataset --output ./augmentation --concept all --use_rembg
"""

import argparse
import glob
import os
from PIL import Image, ImageEnhance, ImageOps

# rembg 패키지 가용성 체크
HAS_REMBG = False
try:
    from rembg import remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

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

# 배경 제거를 적용하지 않을 서브젝트 (풍경/전경 전체가 과제인 경우)
SKIP_REMBG_CONCEPTS = ["scene_waterfall"]


def fit_and_pad(img: Image.Image, target_size: int = 512, bg_color=(255, 255, 255)) -> Image.Image:
    """이미지 비율을 유지하면서 target_size x target_size 중앙에 패딩 배치"""
    img = ImageOps.exif_transpose(img).convert("RGB")
    
    # 비율 유지 축소/확대
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    
    # 중앙 패딩 캔버스 생성
    new_img = Image.new("RGB", (target_size, target_size), bg_color)
    paste_x = (target_size - img.width) // 2
    paste_y = (target_size - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def augment_image(
    image_path: str,
    output_dir: str,
    base_name: str,
    concept_name: str,
    target_size: int = 512,
    use_rembg: bool = False
):
    """단일 이미지에 대한 정밀 증강 수행"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        raw_img = Image.open(image_path)
    except Exception as e:
        print(f"  [오류] 이미지 로드 실패 ({image_path}): {e}")
        return 0

    # 1. 표준 중앙 패딩 512x512 (기본 변환)
    img_standard = fit_and_pad(raw_img, target_size=target_size)
    save_path_std = os.path.join(output_dir, f"{base_name}_std.png")
    img_standard.save(save_path_std)
    count = 1

    # 2. 좌우 반전 (Horizontal Flip)
    img_flip = img_standard.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    save_path_flip = os.path.join(output_dir, f"{base_name}_flip.png")
    img_flip.save(save_path_flip)
    count += 1

    # 3. 미세 조명/대비 변형 (Light Contrast Adjust: 1.1x)
    enhancer = ImageEnhance.Contrast(img_standard)
    img_contrast = enhancer.enhance(1.1)
    save_path_contrast = os.path.join(output_dir, f"{base_name}_contrast.png")
    img_contrast.save(save_path_contrast)
    count += 1

    # 4. 배경 제거 (Background Removal - rembg)
    if use_rembg and HAS_REMBG and (concept_name not in SKIP_REMBG_CONCEPTS):
        try:
            # 배경 제거 (RGBA)
            raw_rgba = ImageOps.exif_transpose(raw_img).convert("RGBA")
            nobg_rgba = remove(raw_rgba)
            
            # 512x512 중앙 배치 (흰색 배경)
            nobg_rgba.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            nobg_canvas = Image.new("RGBA", (target_size, target_size), (255, 255, 255, 255))
            paste_x = (target_size - nobg_rgba.width) // 2
            paste_y = (target_size - nobg_rgba.height) // 2
            nobg_canvas.paste(nobg_rgba, (paste_x, paste_y), mask=nobg_rgba)
            
            img_nobg = nobg_canvas.convert("RGB")
            save_path_nobg = os.path.join(output_dir, f"{base_name}_nobg.png")
            img_nobg.save(save_path_nobg)
            count += 1

            # 배경 제거 + 좌우 반전
            img_nobg_flip = img_nobg.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            save_path_nobg_flip = os.path.join(output_dir, f"{base_name}_nobg_flip.png")
            img_nobg_flip.save(save_path_nobg_flip)
            count += 1

        except Exception as e:
            print(f"  [경고] rembg 배경 제거 수행 중 문제 발생 ({base_name}): {e}")

    return count


def process_concept(
    dataset_dir: str,
    output_dir: str,
    concept: str,
    target_size: int = 512,
    use_rembg: bool = False
):
    concept_in_dir = os.path.join(dataset_dir, concept)
    concept_out_dir = os.path.join(output_dir, concept)

    if not os.path.exists(concept_in_dir):
        print(f"[경고] 존재하지 않는 서브젝트 폴더입니다: {concept_in_dir}")
        return

    image_paths = sorted(
        glob.glob(os.path.join(concept_in_dir, "*.png")) +
        glob.glob(os.path.join(concept_in_dir, "*.jpg")) +
        glob.glob(os.path.join(concept_in_dir, "*.jpeg"))
    )

    print(f"\n▶ [{concept}] 증강 작업 시작 (원본: {len(image_paths)}장)")
    total_aug_count = 0

    for idx, img_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        file_key = f"{idx:02d}_{base_name}"
        aug_count = augment_image(
            image_path=img_path,
            output_dir=concept_out_dir,
            base_name=file_key,
            concept_name=concept,
            target_size=target_size,
            use_rembg=use_rembg
        )
        total_aug_count += aug_count

    print(f"  ✓ [{concept}] 완료! 증강 결과 총 {total_aug_count}장 생성 -> {concept_out_dir}")


def main():
    parser = argparse.ArgumentParser(description="CustomConcept101 Dataset Augmentation Pipeline")
    parser.add_argument("--dataset", type=str, default="./dataset", help="원본 레퍼런스 데이터셋 폴더 경로")
    parser.add_argument("--output", type=str, default="./augmentation", help="증강 이미지 저장 폴더 경로")
    parser.add_argument(
        "--concept",
        type=str,
        default="all",
        help="증강할 서브젝트명 ('all' 지정 시 10개 전체, 또는 'actionfigure_2' 등 특정 서브젝트 지정)"
    )
    parser.add_argument("--target_size", type=int, default=512, help="목표 해상도 (기본 512)")
    parser.add_argument(
        "--use_rembg",
        action="store_true",
        help="rembg 설치 시 배경 제거 증강 포함 (기본 미사용, 옵션 추가 시 사용)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("   CustomConcept101 데이터 증강 파이프라인 (Augmentation)")
    print("=" * 60)
    print(f"- 입력 경로: {args.dataset}")
    print(f"- 출력 경로: {args.output}")
    print(f"- 대상 서브젝트: {args.concept}")
    print(f"- Target Resolution: {args.target_size}x{args.target_size}")
    print(f"- rembg 배경 제거 사용 여부: {args.use_rembg and HAS_REMBG}")
    if args.use_rembg and not HAS_REMBG:
        print("  ⚠️ rembg 패키지가 설치되어 있지 않아 일반 증강(Crop/Flip/Contrast)만 진행됩니다.")
        print("  💡 배경 제거 포함을 원하시면 'pip install rembg' 후 재실행하세요.")

    if args.concept == "all":
        target_concepts = ALL_CONCEPTS
    else:
        if args.concept not in ALL_CONCEPTS:
            print(f"[경고] '{args.concept}'는 정의된 10개 서브젝트 목록에 없습니다. 폴더명을 직접 검색합니다.")
        target_concepts = [args.concept]

    for concept in target_concepts:
        process_concept(
            dataset_dir=args.dataset,
            output_dir=args.output,
            concept=concept,
            target_size=args.target_size,
            use_rembg=args.use_rembg
        )

    print("\n" + "=" * 60)
    print(f"🎉 모든 증강 작업이 완료되었습니다! 결과 폴더: {os.path.abspath(args.output)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
