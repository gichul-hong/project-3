import shutil
"""
Exp-12 Multi-Condition Offline Reranker & Benchmark
Sweeps across:
(a) Current: W_T=1.5, W_I=0.9
(b) 1:1 Aligned: W_T=1.0, W_I=1.0
(c) 1:1 Aligned + border_white_frac penalty
"""

import glob
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

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

def _unwrap(x):
    return x if isinstance(x, torch.Tensor) else x.pooler_output

def compute_border_white_frac(img: Image.Image, thresh: int = 240, border_ratio: float = 0.12) -> float:
    """Calculates the fraction of nearly pure white pixels along the outer image border."""
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    bh, bw = max(1, int(h * border_ratio)), max(1, int(w * border_ratio))
    
    top = arr[:bh, :, :]
    bottom = arr[-bh:, :, :]
    left = arr[:, :bw, :]
    right = arr[:, -bw:, :]
    
    def _is_white(chunk):
        return (chunk[:, :, 0] >= thresh) & (chunk[:, :, 1] >= thresh) & (chunk[:, :, 2] >= thresh)
    
    white_cnt = np.sum(_is_white(top)) + np.sum(_is_white(bottom)) + np.sum(_is_white(left)) + np.sum(_is_white(right))
    total_cnt = top.shape[0]*top.shape[1] + bottom.shape[0]*bottom.shape[1] + left.shape[0]*left.shape[1] + right.shape[0]*right.shape[1]
    return float(white_cnt / max(1, total_cnt))

def run_rerank(output_best_to_dir=None):
    root = "/content/project-3"
    cands_dir = os.path.join(root, "experiments", "12_balanced_ensemble", "candidates")
    if not os.path.exists(cands_dir):
        print("Candidates directory not ready yet!")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_id).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_id)

    data = {}
    for concept, class_noun in CLASS_PROMPT.items():
        prompts_file = os.path.join(root, "prompt", f"{concept}.txt")
        with open(prompts_file, "r", encoding="utf-8") as f:
            eval_prompts = [l.strip().replace("{}", class_noun) for l in f.readlines() if l.strip()]

        ref_paths = sorted(
            glob.glob(os.path.join(root, "dataset", concept, "*.png")) +
            glob.glob(os.path.join(root, "dataset", concept, "*.jpg")) +
            glob.glob(os.path.join(root, "dataset", concept, "*.jpeg"))
        )
        ref_imgs = [Image.open(p).convert("RGB") for p in ref_paths]

        cand_imgs_per_prompt = []
        white_frac_per_prompt = []
        for pi in range(10):
            c_imgs = []
            c_wf = []
            for ci in range(4):
                p = os.path.join(cands_dir, concept, f"p{pi}_c{ci}.png")
                im = Image.open(p).convert("RGB")
                c_imgs.append(im)
                c_wf.append(compute_border_white_frac(im))
            cand_imgs_per_prompt.append(c_imgs)
            white_frac_per_prompt.append(c_wf)

        b_txt = clip_proc(text=eval_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        te = F.normalize(_unwrap(clip_model.get_text_features(**b_txt)).float(), dim=-1)

        b_ref = clip_proc(images=ref_imgs, return_tensors="pt").to(device)
        ri = F.normalize(_unwrap(clip_model.get_image_features(**b_ref)).float(), dim=-1)

        gi_list = []
        for pi in range(10):
            b_cand = clip_proc(images=cand_imgs_per_prompt[pi], return_tensors="pt").to(device)
            gi = F.normalize(_unwrap(clip_model.get_image_features(**b_cand)).float(), dim=-1)
            gi_list.append(gi)

        data[concept] = {
            "eval_prompts": eval_prompts,
            "te": te,
            "ri": ri,
            "gi_list": gi_list,
            "cand_imgs": cand_imgs_per_prompt,
            "white_frac": white_frac_per_prompt
        }

    configs = [
        ("Current (W_T=1.5, W_I=0.9, No White Penalty)", 1.5, 0.9, False),
        ("1:1 Official Aligned (W_T=1.0, W_I=1.0)", 1.0, 1.0, False),
        ("1:1 Aligned + White Penalty (W_T=1.0, W_I=1.0, WhitePen=1.5)", 1.0, 1.0, True),
        ("Balanced + White Penalty (W_T=1.2, W_I=1.0, WhitePen=1.5)", 1.2, 1.0, True),
    ]

    print("\n" + "=" * 90)
    print(f"{'Configuration':<50} | {'CLIP-T':<10} | {'CLIP-I':<10} | {'Total Score':<12}")
    print("=" * 90)

    best_cfg_name = None
    best_total = -1
    best_selections = {}

    for label, wt, wi, use_white_pen in configs:
        t_means, i_means = [], []
        cfg_selections = {}

        for concept in CLASS_PROMPT.keys():
            c_data = data[concept]
            te = c_data["te"]
            ri = c_data["ri"]
            chosen_embs = []
            c_t, c_i = [], []
            concept_picks = []

            for pi in range(10):
                gi = c_data["gi_list"][pi]
                s_t = gi @ te[pi]
                s_i = (gi @ ri.T).mean(dim=1)
                s_dup = (gi @ ri.T).max(dim=1).values

                score = wt * s_t + wi * s_i - 1.0 * (s_dup - 0.92).clamp(min=0)
                if use_white_pen:
                    wf = torch.tensor(c_data["white_frac"][pi], device=device, dtype=torch.float32)
                    score = score - 1.5 * (wf - 0.20).clamp(min=0)

                if chosen_embs:
                    prev = torch.stack(chosen_embs)
                    score = score - 0.35 * (gi @ prev.T).max(dim=1).values

                b = int(score.argmax().item())
                chosen_embs.append(gi[b])
                c_t.append(float(s_t[b].item()))
                c_i.append(float(s_i[b].item()))
                concept_picks.append(b)

            t_means.append(np.mean(c_t))
            i_means.append(np.mean(c_i))
            cfg_selections[concept] = concept_picks

        avg_t = float(np.mean(t_means))
        avg_i = float(np.mean(i_means))
        total = avg_t + avg_i
        print(f"{label:<50} | {avg_t:.4f}{'':<4} | {avg_i:.4f}{'':<4} | {total:.4f}")

        if total > best_total:
            best_total = total
            best_cfg_name = label
            best_selections = cfg_selections

    print("=" * 90)
    print(f"🌟 Best Configuration: {best_cfg_name} (Total: {best_total:.4f})")

    # If requested, apply best selections to destination folder
    if output_best_to_dir and best_selections:
        print(f"\n🚀 Applying best selection ({best_cfg_name}) to: {output_best_to_dir}")
        for concept, picks in best_selections.items():
            out_c_dir = os.path.join(output_best_to_dir, concept)
            os.makedirs(out_c_dir, exist_ok=True)
            for pi, ci in enumerate(picks):
                src = os.path.join(cands_dir, concept, f"p{pi}_c{ci}.png")
                dst = os.path.join(out_c_dir, f"{pi}.png")
                shutil.copy(src, dst)
        print("✓ All 100 optimal images copied!")

if __name__ == "__main__":
    run_rerank(output_best_to_dir="/content/project-3/experiments/12_balanced_ensemble")
