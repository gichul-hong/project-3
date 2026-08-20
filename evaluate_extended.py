"""
Extended & Creative Evaluation Suite for Multi-Subject Customization
---------------------------------------------------------------------
Evaluates customization quality across 4 multi-dimensional axes:
1. Standard CLIP Metrics:
   - CLIP-T (Text-to-Image Alignment)
   - CLIP-I (Subject Identity Preservation)
2. Structural & Semantic Consistency:
   - DINOv2-I (Self-Supervised Feature Cosine Similarity - robust to texture/lighting variations)
3. 4-Axis Generalization Taxonomy:
   - Style Transfer
   - Attribute Binding
   - Scene Composition
   - Action Dynamics
4. Generative Diversity:
   - Intra-Concept Pairwise Distance (Diversity score preventing mode collapse)
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Any

from dotenv import load_dotenv
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel

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

PROMPT_TAXONOMY_TAGS = {
    "actionfigure_2": {
        0: "Baseline Instance",
        1: "Scene Composition",
        2: "Scene Composition",
        3: "Scene Composition",
        4: "Scene Composition",
        5: "Scene Composition",
        6: "Action Dynamics",
        7: "Action Dynamics",
        8: "Attribute Binding",
        9: "Attribute Binding",
    },
    "decoritems_woodenpot": {
        0: "Baseline Instance",
        1: "Attribute Binding",
        2: "Scene Composition",
        3: "Scene Composition",
        4: "Scene Composition",
        5: "Attribute Binding",
        6: "Attribute Binding",
        7: "Attribute Binding",
        8: "Scene Composition",
        9: "Attribute Binding",
    },
    "furniture_sofa2": {
        0: "Baseline Instance",
        1: "Scene Composition",
        2: "Scene Composition",
        3: "Scene Composition",
        4: "Scene Composition",
        5: "Scene Composition",
        6: "Attribute Binding",
        7: "Attribute Binding",
        8: "Attribute Binding",
        9: "Style Transfer",
    },
    "instrument_music2": {
        0: "Baseline Instance",
        1: "Scene Composition",
        2: "Scene Composition",
        3: "Scene Composition",
        4: "Scene Composition",
        5: "Scene Composition",
        6: "Attribute Binding",
        7: "Attribute Binding",
        8: "Style Transfer",
        9: "Style Transfer",
    },
    "luggage_backpack1": {
        0: "Baseline Instance",
        1: "Scene Composition",
        2: "Scene Composition",
        3: "Scene Composition",
        4: "Scene Composition",
        5: "Scene Composition",
        6: "Attribute Binding",
        7: "Attribute Binding",
        8: "Style Transfer",
        9: "Style Transfer",
    },
    "person_3": {
        0: "Baseline Instance",
        1: "Scene Composition",
        2: "Scene Composition",
        3: "Scene Composition",
        4: "Scene Composition",
        5: "Scene Composition",
        6: "Attribute Binding",
        7: "Attribute Binding",
        8: "Attribute Binding",
        9: "Attribute Binding",
    },
    "pet_cat5": {
        0: "Baseline Instance",
        1: "Scene Composition",
        2: "Scene Composition",
        3: "Scene Composition",
        4: "Scene Composition",
        5: "Scene Composition",
        6: "Attribute Binding",
        7: "Attribute Binding",
        8: "Attribute Binding",
        9: "Attribute Binding",
    },
    "scene_waterfall": {
        0: "Baseline Instance",
        1: "Style Transfer",
        2: "Style Transfer",
        3: "Style Transfer",
        4: "Style Transfer",
        5: "Attribute Binding",
        6: "Attribute Binding",
        7: "Attribute Binding",
        8: "Attribute Binding",
        9: "Attribute Binding",
    },
    "transport_tank": {
        0: "Baseline Instance",
        1: "Scene Composition",
        2: "Scene Composition",
        3: "Scene Composition",
        4: "Scene Composition",
        5: "Scene Composition",
        6: "Attribute Binding",
        7: "Attribute Binding",
        8: "Attribute Binding",
        9: "Attribute Binding",
    },
    "wearable_jacket1": {
        0: "Baseline Instance",
        1: "Scene Composition",
        2: "Scene Composition",
        3: "Scene Composition",
        4: "Scene Composition",
        5: "Scene Composition",
        6: "Attribute Binding",
        7: "Attribute Binding",
        8: "Style Transfer",
        9: "Style Transfer",
    },
}


class ExtendedEvaluator:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("📊 Extended Multi-Metric Evaluator 로딩 중...")

        # 1. CLIP-L/14 로드
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # 2. DINOv2 ViT-S/14 로드 (Structural & Self-Supervised Alignment)
        try:
            self.dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device).eval()
            self.dino_transform = T.Compose([
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.has_dino = True
            print("✓ DINOv2-ViT-S/14 로드 완료!")
        except Exception as e:
            print(f"⚠️ DINOv2 로드 실패 (스킵): {e}")
            self.has_dino = False

    def _extract_tensor(self, x: Any) -> torch.Tensor:
        if hasattr(x, "pooler_output") and x.pooler_output is not None:
            return x.pooler_output
        if hasattr(x, "image_embeds") and x.image_embeds is not None:
            return x.image_embeds
        if hasattr(x, "text_embeds") and x.text_embeds is not None:
            return x.text_embeds
        return x

    @torch.no_grad()
    def compute_clip_image_features(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        feats = self._extract_tensor(self.clip_model.get_image_features(**inputs))
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def compute_clip_text_features(self, texts: List[str]) -> torch.Tensor:
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        feats = self._extract_tensor(self.clip_model.get_text_features(**inputs))
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def compute_dino_features(self, images: List[Image.Image]) -> torch.Tensor:
        if not self.has_dino:
            return None
        tensors = torch.stack([self.dino_transform(img.convert("RGB")) for img in images]).to(self.device)
        feats = self.dinov2_model(tensors)
        return F.normalize(feats, dim=-1)

    def evaluate_experiment(
        self,
        exp_dir: str,
        data_base_dir: str = "./data",
    ) -> Dict:
        """한 실험 디렉토리의 전체 10개 서브젝트에 대한 다차원 평가 수행"""
        print(f"\n🔍 [Extended Eval] '{exp_dir}' 정밀 다차원 평가 시작...")
        concept_dirs = sorted([d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d)) and not d.startswith(".")])

        results = {}
        all_clip_t = []
        all_clip_i = []
        all_dino_i = []
        taxonomy_scores = {
            "Baseline Instance": {"clip_t": [], "clip_i": [], "dino_i": []},
            "Scene Composition": {"clip_t": [], "clip_i": [], "dino_i": []},
            "Attribute Binding": {"clip_t": [], "clip_i": [], "dino_i": []},
            "Style Transfer": {"clip_t": [], "clip_i": [], "dino_i": []},
            "Action Dynamics": {"clip_t": [], "clip_i": [], "dino_i": []},
        }

        for concept in concept_dirs:
            gen_dir = os.path.join(exp_dir, concept)
            ref_dir = os.path.join(data_base_dir, concept)

            gen_img_paths = sorted(glob.glob(os.path.join(gen_dir, "*.png")) + glob.glob(os.path.join(gen_dir, "*.jpg")))
            if not gen_img_paths:
                continue

            ref_img_paths = sorted(glob.glob(os.path.join(ref_dir, "*.png")) + glob.glob(os.path.join(ref_dir, "*.jpg")) + glob.glob(os.path.join(ref_dir, "*.jpeg")))
            if not ref_img_paths:
                continue

            # 1. 이미지 로드
            gen_images = [Image.open(p).convert("RGB") for p in gen_img_paths]
            ref_images = [Image.open(p).convert("RGB") for p in ref_img_paths]

            # 2. 텍스트 프롬프트 로드
            prompts = []
            prompt_file = os.path.join(ref_dir, "prompts.json")
            txt_file = os.path.join("prompt", f"{concept}.txt")
            if os.path.exists(prompt_file):
                with open(prompt_file) as f:
                    prompts = json.load(f)
            elif os.path.exists(txt_file):
                with open(txt_file, "r", encoding="utf-8") as tf:
                    class_w = CLASS_PROMPT.get(concept, concept)
                    prompts = [l.strip().replace("{}", class_w) for l in tf if l.strip()]
            else:
                prompts = [f"a photo of {concept}"] * len(gen_images)

            # 3. 임베딩 계산
            gen_clip_img = self.compute_clip_image_features(gen_images)
            ref_clip_img = self.compute_clip_image_features(ref_images)

            # CLIP-T 프롬프트 정제 (sks 제거 후 측정)
            clean_prompts = [p.replace("sks ", "").replace("sks", "") for p in prompts[:len(gen_images)]]
            clip_text = self.compute_clip_text_features(clean_prompts)

            # CLIP-T: pairwise text-image similarity
            t2i_scores = (clip_text * gen_clip_img).sum(dim=-1).cpu().numpy().tolist()

            # CLIP-I: reference 평균과의 cosine similarity
            ref_clip_mean = F.normalize(ref_clip_img.mean(dim=0, keepdim=True), dim=-1)
            i2i_scores = (gen_clip_img * ref_clip_mean).sum(dim=-1).cpu().numpy().tolist()

            # DINO-I
            dino_scores = []
            if self.has_dino:
                gen_dino = self.compute_dino_features(gen_images)
                ref_dino = self.compute_dino_features(ref_images)
                ref_dino_mean = F.normalize(ref_dino.mean(dim=0, keepdim=True), dim=-1)
                dino_scores = (gen_dino * ref_dino_mean).sum(dim=-1).cpu().numpy().tolist()

            # Concept Diversity (L2 pairwise distance between generated latents)
            pairwise_dists = torch.cdist(gen_clip_img, gen_clip_img, p=2).cpu().numpy()
            n = len(gen_images)
            diversity = float(pairwise_dists.sum() / (n * (n - 1))) if n > 1 else 0.0

            concept_tags = PROMPT_TAXONOMY_TAGS.get(concept, {})

            # Taxonomy별 집계
            for idx in range(len(gen_images)):
                tag = concept_tags.get(idx, "Scene Composition")
                if tag in taxonomy_scores:
                    taxonomy_scores[tag]["clip_t"].append(t2i_scores[idx])
                    taxonomy_scores[tag]["clip_i"].append(i2i_scores[idx])
                    if dino_scores:
                        taxonomy_scores[tag]["dino_i"].append(dino_scores[idx])

            avg_t = float(np.mean(t2i_scores))
            avg_i = float(np.mean(i2i_scores))
            avg_dino = float(np.mean(dino_scores)) if dino_scores else 0.0

            all_clip_t.append(avg_t)
            all_clip_i.append(avg_i)
            if dino_scores:
                all_dino_i.append(avg_dino)

            results[concept] = {
                "clip_t": round(avg_t, 4),
                "clip_i": round(avg_i, 4),
                "dino_i": round(avg_dino, 4),
                "diversity": round(diversity, 4),
                "total": round(avg_t + avg_i, 4),
            }

        # Taxonomy 평균 계산
        taxonomy_summary = {}
        for cat, vals in taxonomy_scores.items():
            if vals["clip_t"]:
                taxonomy_summary[cat] = {
                    "count": len(vals["clip_t"]),
                    "clip_t": round(float(np.mean(vals["clip_t"])), 4),
                    "clip_i": round(float(np.mean(vals["clip_i"])), 4),
                    "dino_i": round(float(np.mean(vals["dino_i"])), 4) if vals["dino_i"] else 0.0,
                }

        total_summary = {
            "clip_t": round(float(np.mean(all_clip_t)), 4),
            "clip_i": round(float(np.mean(all_clip_i)), 4),
            "dino_i": round(float(np.mean(all_dino_i)), 4) if all_dino_i else 0.0,
            "total_clip": round(float(np.mean(all_clip_t)) + float(np.mean(all_clip_i)), 4),
        }

        full_eval = {
            "experiment": os.path.basename(exp_dir),
            "summary": total_summary,
            "taxonomy_breakdown": taxonomy_summary,
            "per_concept": results,
        }

        # 저장
        out_path = os.path.join(exp_dir, "extended_eval.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(full_eval, f, indent=2, ensure_ascii=False)

        print(f"✓ Extended Evaluation 저장 완료: {out_path}")
        print(f"  [Summary] CLIP-T: {total_summary['clip_t']:.4f}, CLIP-I: {total_summary['clip_i']:.4f}, DINO-I: {total_summary['dino_i']:.4f}, Total: {total_summary['total_clip']:.4f}")
        return full_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="all", help="특정 실험 디렉토리 또는 'all'")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    evaluator = ExtendedEvaluator()

    if args.exp_dir == "all":
        exp_dirs = sorted(glob.glob("experiments/*"))
    else:
        exp_dirs = [args.exp_dir]

    all_comparisons = {}
    for ed in exp_dirs:
        if os.path.isdir(ed):
            res = evaluator.evaluate_experiment(ed, args.data_dir)
            all_comparisons[os.path.basename(ed)] = res

    # 전체 비교 요약 저장
    comp_file = "experiments/EXTENDED_COMPARISON.json"
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(all_comparisons, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 전체 실험 다차원 비교 분석 완료: {comp_file}")


if __name__ == "__main__":
    main()
