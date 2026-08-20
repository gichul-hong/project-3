import glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

root = "/content/project-3"
exp_dir = os.path.join(root, "experiments", "11_best_of_n_ensemble")
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_id).to(device).eval()
clip_proc = CLIPProcessor.from_pretrained(clip_id)

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

print("=" * 80)
print("📊 [Exp-11 Official CLIP-ViT-B/32 Per-Concept Quantitative Results]")
print("=" * 80)

summary = {"per_concept": {}, "average": {}}
t_list, i_list = [], []

for concept, class_noun in CLASS_PROMPT.items():
    prompts_file = os.path.join(root, "prompt", f"{concept}.txt")
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [l.strip().replace("{}", class_noun) for l in f.readlines() if l.strip()]

    gen_imgs = [Image.open(os.path.join(exp_dir, concept, f"{i}.png")).convert("RGB") for i in range(len(prompts))]
    ref_paths = sorted(glob.glob(os.path.join(root, "dataset", concept, "*.*")))
    ref_imgs = [Image.open(p).convert("RGB") for p in ref_paths]

    b_t = clip_proc(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    te = F.normalize(_unwrap(clip_model.get_text_features(**b_t)).float(), dim=-1)

    b_g = clip_proc(images=gen_imgs, return_tensors="pt").to(device)
    gi = F.normalize(_unwrap(clip_model.get_image_features(**b_g)).float(), dim=-1)

    b_r = clip_proc(images=ref_imgs, return_tensors="pt").to(device)
    ri = F.normalize(_unwrap(clip_model.get_image_features(**b_r)).float(), dim=-1)

    score_t = float(F.cosine_similarity(gi, te).mean().item())
    score_i = float((gi @ ri.T).mean().item())

    summary["per_concept"][concept] = {"clip_t": score_t, "clip_i": score_i, "total": score_t + score_i}
    t_list.append(score_t)
    i_list.append(score_i)

    print(f"• {concept:<22}: CLIP-T = {score_t:.4f} | CLIP-I = {score_i:.4f} | Total = {score_t + score_i:.4f}")

mean_t = float(np.mean(t_list))
mean_i = float(np.mean(i_list))
total_mean = mean_t + mean_i
summary["average"] = {"clip_t": mean_t, "clip_i": mean_i, "total": total_mean}

print("=" * 80)
print(f"🏆 [Exp-11 Official Summary] CLIP-T: {mean_t:.4f} | CLIP-I: {mean_i:.4f} | Total: {total_mean:.4f}")
print("=" * 80)

with open(os.path.join(exp_dir, "official_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
