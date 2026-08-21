import glob
import json
import os

root = r"C:\hong\project-3"
exp_dir = os.path.join(root, "experiments", "13_sota_ensemble")

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

best_per_concept = []

for concept, class_noun in CLASS_PROMPT.items():
    prompts_file = os.path.join(root, "prompt", f"{concept}.txt")
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [l.strip().replace("{}", class_noun) for l in f.readlines() if l.strip()]

    sel_file = os.path.join(exp_dir, f"selection_{concept}.json")
    with open(sel_file, "r", encoding="utf-8") as f:
        sel_data = json.load(f)

    prompt_scores = []
    for item in sel_data:
        p_idx = item["prompt_idx"]
        t = item["clip_t"]
        i = item["clip_i"]
        tot = t + i
        p_text = prompts[p_idx]
        img_path = os.path.join(exp_dir, concept, f"{p_idx}.png").replace("\\", "/")
        prompt_scores.append({
            "concept": concept,
            "prompt_idx": p_idx,
            "prompt": p_text,
            "clip_t": t,
            "clip_i": i,
            "total": tot,
            "img_path": img_path
        })

    # 프롬프트 0(단순 흰배경 누끼 인스턴스)보다 배경/행동 합성이 들어간 1~9 중 최고점
    scene_candidates = [x for x in prompt_scores if x["prompt_idx"] > 0]
    best_scene = max(scene_candidates, key=lambda x: x["total"])

    best_per_concept.append({
        "concept": concept,
        "best_scene": best_scene,
        "all_scores": prompt_scores
    })

# 10개 컨셉을 최고 장면 점수 순으로 정렬
best_per_concept.sort(key=lambda x: x["best_scene"]["total"], reverse=True)

print("=" * 85)
print("🏆 [Exp-13 SOTA Ensemble] 10개 서브젝트별 최고 퀄리티 장면 랭킹 (Top 8 선정용)")
print("=" * 85)

for rank, item in enumerate(best_per_concept, 1):
    c = item["concept"]
    b = item["best_scene"]
    status = "★ TOP 8 선정" if rank <= 8 else "  (차순위)"
    print(f"{rank:2d}. [{c:<20}] {status}")
    print(f"    • Prompt #{b['prompt_idx']}: \"{b['prompt']}\"")
    print(f"    • Score: Total = {b['total']:.4f} (CLIP-T: {b['clip_t']:.4f}, CLIP-I: {b['clip_i']:.4f})")
    print(f"    • Image: {b['img_path']}")
    print("-" * 85)
