"""
CustomConcept101 Dataset Augmentation Script (v2 - rembg + caption + selective flip)
----------------------------------------------------------------------
사용법:
  1) 전체 카테고리(10개) 증강 (rembg 포함):
     python augment_dataset.py --dataset ./dataset --output ./augmentation --concept all --use_rembg

  2) 특정 카테고리만 증강:
     python augment_dataset.py --dataset ./dataset --output ./augmentation --concept actionfigure_2 --use_rembg

  3) 배경 제거 없이 기본 증강만:
     python augment_dataset.py --dataset ./dataset --output ./augmentation --concept all
"""

import argparse
import glob
import json
import os
from PIL import Image, ImageEnhance, ImageOps

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

SKIP_REMBG_CONCEPTS = ["scene_waterfall"]

SKIP_FLIP_CONCEPTS = [
    "instrument_music2",
    "luggage_backpack1",
    "transport_tank",
    "wearable_jacket1",
]


def fit_and_pad(img: Image.Image, target_size: int = 512, bg_color=(255, 255, 255)) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
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
    use_rembg: bool = False,
    trigger_token: str = "sks",
):
    os.makedirs(output_dir, exist_ok=True)
    class_word = CLASS_PROMPT.get(concept_name, concept_name)
    caption = f"a photo of {trigger_token} {class_word}"
    skip_flip = concept_name in SKIP_FLIP_CONCEPTS
    skip_rembg = concept_name in SKIP_REMBG_CONCEPTS
    metadata = []

    try:
        raw_img = Image.open(image_path)
    except Exception as e:
        print(f"  [오류] 이미지 로드 실패 ({image_path}): {e}")
        return metadata

    def _save(name_suffix, img):
        fname = f"{base_name}{name_suffix}.png"
        img.save(os.path.join(output_dir, fname))
        cap_path = os.path.join(output_dir, f"{base_name}{name_suffix}.txt")
        with open(cap_path, "w", encoding="utf-8") as f:
            f.write(caption)
        metadata.append({"file_name": fname, "text": caption})

    img_std = fit_and_pad(raw_img, target_size=target_size)
    _save("_std", img_std)

    if not skip_flip:
        img_flip = img_std.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        _save("_flip", img_flip)

    enhancer_b = ImageEnhance.Brightness(img_std)
    enhancer_c = ImageEnhance.Contrast(img_std)
    img_light = enhancer_b.enhance(1.05)
    img_light = enhancer_c.enhance(1.10)
    _save("_light", img_light)

    if use_rembg and HAS_REMBG and not skip_rembg:
        try:
            raw_rgba = ImageOps.exif_transpose(raw_img).convert("RGBA")
            nobg_rgba = remove(raw_rgba)
            nobg_rgba.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            nobg_canvas = Image.new("RGBA", (target_size, target_size), (255, 255, 255, 255))
            px = (target_size - nobg_rgba.width) // 2
            py = (target_size - nobg_rgba.height) // 2
            nobg_canvas.paste(nobg_rgba, (px, py), mask=nobg_rgba)
            img_nobg = nobg_canvas.convert("RGB")
            _save("_nobg", img_nobg)

            if not skip_flip:
                img_nobg_flip = img_nobg.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                _save("_nobg_flip", img_nobg_flip)
        except Exception as e:
            print(f"  [경고] rembg 배경 제거 실패 ({base_name}): {e}")

    return metadata


def process_concept(
    dataset_dir: str,
    output_dir: str,
    concept: str,
    target_size: int = 512,
    use_rembg: bool = False,
    trigger_token: str = "sks",
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

    print(f"\n▶ [{concept}] 증강 시작 (원본: {len(image_paths)}장)")
    all_metadata = []

    for idx, img_path in enumerate(image_paths):
        base = os.path.splitext(os.path.basename(img_path))[0]
        file_key = f"{idx:02d}_{base}"
        meta = augment_image(
            image_path=img_path,
            output_dir=concept_out_dir,
            base_name=file_key,
            concept_name=concept,
            target_size=target_size,
            use_rembg=use_rembg,
            trigger_token=trigger_token,
        )
        all_metadata.extend(meta)

    jsonl_path = os.path.join(concept_out_dir, "metadata.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in all_metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    skip_flip = concept in SKIP_FLIP_CONCEPTS
    skip_rembg = concept in SKIP_REMBG_CONCEPTS
    tags = []
    tags.append(f"flip={'no' if skip_flip else 'yes'}")
    tags.append(f"rembg={'no' if (not use_rembg or not HAS_REMBG or skip_rembg) else 'yes'}")
    print(f"  ✓ [{concept}] 완료! 증강 {len(all_metadata)}장, 캡션: \"{all_metadata[0]['text'] if all_metadata else 'N/A'}\"  [{', '.join(tags)}]")


def main():
    p = argparse.ArgumentParser(description="CustomConcept101 Dataset Augmentation (v2)")
    p.add_argument("--dataset", default="./dataset", help="원본 레퍼런스 데이터셋 폴더")
    p.add_argument("--output", default="./augmentation", help="증강 이미지 저장 폴더")
    p.add_argument("--concept", default="all", help="서브젝트명 또는 'all'")
    p.add_argument("--target_size", type=int, default=512, help="목표 해상도 (기본 512)")
    p.add_argument("--use_rembg", action="store_true", help="rembg 배경 제거 증강 포함")
    p.add_argument("--trigger_token", default="sks", help="LoRA 캡션용 트리거 토큰 (기본: sks)")
    args = p.parse_args()

    print("=" * 60)
    print("   CustomConcept101 데이터 증강 v2 (rembg + caption + selective flip)")
    print("=" * 60)
    print(f"  입력: {args.dataset}  |  출력: {args.output}  |  대상: {args.concept}")
    print(f"  해상도: {args.target_size}x{args.target_size}  |  트리거 토큰: {args.trigger_token}")
    print(f"  rembg 사용: {args.use_rembg and HAS_REMBG}")
    if args.use_rembg and not HAS_REMBG:
        print("  ⚠️ rembg 미설치 → pip install rembg[cpu]")

    targets = ALL_CONCEPTS if args.concept == "all" else [args.concept]

    for concept in targets:
        process_concept(
            dataset_dir=args.dataset,
            output_dir=args.output,
            concept=concept,
            target_size=args.target_size,
            use_rembg=args.use_rembg,
            trigger_token=args.trigger_token,
        )

    print("\n" + "=" * 60)
    print(f"  전체 증강 완료  →  {os.path.abspath(args.output)}")
    print("=" * 60)


if __name__ == "__main__":
    main()