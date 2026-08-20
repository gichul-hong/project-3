import os, glob, json, sys
import numpy as np

root = "/content/project-3"
exp_dir = os.path.join(root, "experiments", "11_best_of_n_ensemble")
eval_script = os.path.join(root, "evaluation.py")

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

print("=" * 80)
print("📊 Exp-11: Official evaluation.py Execution on all 10 Concepts")
print("=" * 80)

# Move selection.json out of concept image folders to avoid Image.open error
for c in CLASS_PROMPT.keys():
    s_json = os.path.join(exp_dir, c, "selection.json")
    if os.path.exists(s_json):
        os.rename(s_json, os.path.join(exp_dir, f"selection_{c}.json"))

# Run official evaluation
for c in CLASS_PROMPT.keys():
    c_dir = os.path.join(exp_dir, c)
    cmd = f"{sys.executable} {eval_script} --dataset {os.path.join(root, 'dataset')} --prompts {os.path.join(root, 'prompt')} --concept {c} --images {c_dir}"
    os.system(cmd)
