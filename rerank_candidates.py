"""
Pareto Multi-Objective Re-ranking Explorer
Evaluates different (W_T, W_I) combinations across all 400 existing candidate images.
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

def main():
    root = "/content/project-3"
    cands_dir = os.path.join(root, "experiments", "11_best_of_n_ensemble", "candidates")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_id).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_id)

    # Precompute all embeddings
    data = {}
    for concept, class_noun in CLASS_PROMPT.items():
        prompts_file = os.path.join(root, "prompt", f"{concept}.txt")
        with open(prompts_file, "r", encoding="utf-8") as f:
            eval_prompts = [l.strip().replace("{}", class_noun) for l in f.readlines() if l.strip()]

        ref_paths = sorted(glob.glob(os.path.join(root, "dataset", concept, "*.*")))
        ref_imgs = [Image.open(p).convert("RGB") for p in ref_paths]

        # candidate images for each prompt (4 cands each)
        cand_imgs_per_prompt = []
        for pi in range(10):
            c_imgs = []
            for ci in range(4):
                p = os.path.join(cands_dir, concept, f"p{pi}_c{ci}.png")
                c_imgs.append(Image.open(p).convert("RGB"))
            cand_imgs_per_prompt.append(c_imgs)

        # encode
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
            "cand_imgs": cand_imgs_per_prompt
        }

    weight_configs = [
        ("Current (W_T=1.0, W_I=1.2)", 1.0, 1.2),
        ("Equal (W_T=1.0, W_I=1.0)", 1.0, 1.0),
        ("Balanced (W_T=1.5, W_I=0.8)", 1.5, 0.8),
        ("Text-Focused (W_T=2.0, W_I=0.6)", 2.0, 0.6),
        ("Text-Dominant (W_T=2.5, W_I=0.5)", 2.5, 0.5),
    ]

    print("=" * 85)
    print(f"{'Configuration':<35} | {'CLIP-T':<10} | {'CLIP-I':<10} | {'Total Score':<12}")
    print("=" * 85)

    best_cfg = None
    best_total = -1

    for label, wt, wi in weight_configs:
        t_means, i_means = [], []
        for concept in CLASS_PROMPT.keys():
            c_data = data[concept]
            te = c_data["te"]
            ri = c_data["ri"]
            chosen_embs = []
            c_t, c_i = [], []

            for pi in range(10):
                gi = c_data["gi_list"][pi]
                s_t = gi @ te[pi]
                s_i = (gi @ ri.T).mean(dim=1)
                s_dup = (gi @ ri.T).max(dim=1).values

                score = wt * s_t + wi * s_i - 1.0 * (s_dup - 0.92).clamp(min=0)
                if chosen_embs:
                    prev = torch.stack(chosen_embs)
                    score = score - 0.35 * (gi @ prev.T).max(dim=1).values

                b = int(score.argmax().item())
                chosen_embs.append(gi[b])
                c_t.append(float(s_t[b].item()))
                c_i.append(float(s_i[b].item()))

            t_means.append(np.mean(c_t))
            i_means.append(np.mean(c_i))

        avg_t = float(np.mean(t_means))
        avg_i = float(np.mean(i_means))
        total = avg_t + avg_i
        print(f"{label:<35} | {avg_t:.4f}{'':<4} | {avg_i:.4f}{'':<4} | {total:.4f}")

if __name__ == "__main__":
    main()
