"""
Experiment & Image Comparison Dashboard Generator (Dual-View: Concept Matrix & Exp 100-Gallery)
------------------------------------------------------------------------------------------------
1. Concept Matrix View: 컨셉별로 Exp-01 ~ Exp-14 실험 간 10개 프롬프트 생성 결과 비교
2. Experiment 100-Image Gallery View: 특정 실험을 선택하여 10개 서브젝트 × 10개 프롬프트 = 100장의 대형 고화질 카드 뷰 제공
"""

import argparse
import glob
import json
import os
import http.server
import socketserver
from PIL import Image

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

EXP_METADATA = {
    "01_rf_inversion_baseline": {"name": "Exp-01: Baseline RF-Inversion", "color": "#94a3b8", "tag": "ODE Inversion (No LoRA)"},
    "03_lora_augmented": {"name": "Exp-03: Augmented SD3.5 LoRA", "color": "#38bdf8", "tag": "LoRA R16 (Pure Text)"},
    "04_lora_rf_hybrid": {"name": "Exp-04: LoRA + RF Hybrid", "color": "#a855f7", "tag": "LoRA + 1st Euler ODE"},
    "05_lora_hq": {"name": "Exp-05: LoRA High-Quality", "color": "#ec4899", "tag": "LoRA R64 (T5-XXL 1k Steps)"},
    "06_hybrid_adaptive": {"name": "Exp-06: Hybrid Multi-Ref Adaptive", "color": "#f59e0b", "tag": "Multi-ref Latent Avg + Cosine eta"},
    "07_heun_custom_neg": {"name": "Exp-07: Heun 50-Step Custom Neg", "color": "#10b981", "tag": "Heun 2nd-order ODE (50 Steps)"},
    "08_dreambooth_prior_loss": {"name": "Exp-08: True DreamBooth Prior Loss", "color": "#6366f1", "tag": "Dual Flow Loss (lambda=0.3)"},
    "09_subject_adaptive_routing": {"name": "Exp-09: Subject Dynamic Routing", "color": "#e11d48", "tag": "Adaptive Routing + Prompt Detail"},
    "11_best_of_n_ensemble": {"name": "Exp-11: Best-of-N Precision Ensemble", "color": "#8b5cf6", "tag": "4 Cands + CLIP MMR Selector"},
    "12_balanced_ensemble": {"name": "Exp-12: Balanced SOTA Ensemble", "color": "#06b6d4", "tag": "Natural Ref + Full Scene Context"},
    "13_sota_ensemble": {"name": "Exp-13: Ultimate SOTA Ensemble", "color": "#10b981", "tag": "Crop Ref + 1:1 Metric + White Guard"},
    "14_extreme_prompt_align": {"name": "Exp-14: Extreme Prompt Alignment", "color": "#f97316", "tag": "CFG 7.5 + Pure CLIP-T Maximizer"},
}


def get_image_info(img_path, root_dir):
    """이미지 상대 경로, 해상도, 용량 추출"""
    rel_path = os.path.relpath(img_path, root_dir).replace("\\", "/")
    file_size_kb = round(os.path.getsize(img_path) / 1024, 1)
    filename = os.path.basename(img_path)
    
    try:
        with Image.open(img_path) as img:
            width, height = img.size
    except Exception:
        width, height = 0, 0
        
    return {
        "filename": filename,
        "rel_path": rel_path,
        "width": width,
        "height": height,
        "size_kb": file_size_kb,
    }


def scan_all(root_dir):
    experiments_dir = os.path.join(root_dir, "experiments")
    exp_folders = []
    if os.path.exists(experiments_dir):
        exp_folders = sorted([
            f for f in os.listdir(experiments_dir)
            if os.path.isdir(os.path.join(experiments_dir, f)) and not f.startswith(".")
        ])

    data = {
        "concepts": {},
        "experiments": exp_folders,
        "exp_meta": EXP_METADATA,
        "scores": {},
        "extended_scores": {},
        "selection_meta": {},
    }

    # 실험별 공식 채점, 확장 평가, 선별 메타데이터 스캔
    for exp in exp_folders:
        summary_path = os.path.join(experiments_dir, exp, "eval_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data["scores"][exp] = json.load(f)
            except Exception:
                pass

        ext_path = os.path.join(experiments_dir, exp, "extended_eval.json")
        if os.path.exists(ext_path):
            try:
                with open(ext_path, "r", encoding="utf-8") as f:
                    data["extended_scores"][exp] = json.load(f)
            except Exception:
                pass

        data["selection_meta"][exp] = {}
        for sel_file in glob.glob(os.path.join(experiments_dir, exp, "selection_*.json")):
            c_name = os.path.basename(sel_file).replace("selection_", "").replace(".json", "")
            try:
                with open(sel_file, "r", encoding="utf-8") as f:
                    raw_s = json.load(f)
                    if isinstance(raw_s, list):
                        data["selection_meta"][exp][c_name] = raw_s
                    elif isinstance(raw_s, dict):
                        # convert dict to list
                        converted = []
                        for k, v in raw_s.items():
                            idx_num = int(k.replace("p", "")) if k.replace("p", "").isdigit() else None
                            if idx_num is not None:
                                item = dict(v)
                                item["prompt_idx"] = idx_num
                                converted.append(item)
                        data["selection_meta"][exp][c_name] = converted
            except Exception:
                pass

    # 컨셉별 이미지 및 프롬프트 스캔
    for concept, class_prompt in CLASS_PROMPT.items():
        orig_dir = os.path.join(root_dir, "dataset", concept)
        orig_imgs = []
        if os.path.exists(orig_dir):
            paths = sorted(
                glob.glob(os.path.join(orig_dir, "*.png")) +
                glob.glob(os.path.join(orig_dir, "*.jpg")) +
                glob.glob(os.path.join(orig_dir, "*.jpeg"))
            )
            orig_imgs = [get_image_info(p, root_dir) for p in paths]

        aug_dir = os.path.join(root_dir, "augmentation", concept)
        aug_imgs = []
        if os.path.exists(aug_dir):
            paths = sorted(
                glob.glob(os.path.join(aug_dir, "*.png")) +
                glob.glob(os.path.join(aug_dir, "*.jpg")) +
                glob.glob(os.path.join(aug_dir, "*.jpeg"))
            )
            aug_imgs = [get_image_info(p, root_dir) for p in paths]

        prompt_file = os.path.join(root_dir, "prompt", f"{concept}.txt")
        prompts = []
        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompts = [
                    l.strip().replace("{}", class_prompt)
                    for l in f.readlines() if l.strip()
                ]

        exp_generated = {}
        for exp in exp_folders:
            concept_exp_dir = os.path.join(experiments_dir, exp, concept)
            exp_imgs = {}
            if os.path.exists(concept_exp_dir):
                for idx in range(len(prompts) if prompts else 10):
                    img_p = os.path.join(concept_exp_dir, f"{idx}.png")
                    if not os.path.exists(img_p):
                        img_p = os.path.join(concept_exp_dir, f"{idx}.jpg")
                    if os.path.exists(img_p):
                        exp_imgs[idx] = get_image_info(img_p, root_dir)
            exp_generated[exp] = exp_imgs

        data["concepts"][concept] = {
            "class_prompt": class_prompt,
            "orig_images": orig_imgs,
            "aug_images": aug_imgs,
            "prompts": prompts,
            "exp_generated": exp_generated
        }

    return data


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VERILUX Project-3: Multi-Subject Customization Matrix & 100-View Gallery</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --sidebar-bg: #070a12;
            --card-bg: #131b2e;
            --card-sub-bg: #0e1524;
            --card-border: #1e293b;
            --card-border-hover: #38bdf8;
            --accent-cyan: #38bdf8;
            --accent-purple: #c084fc;
            --accent-green: #4ade80;
            --accent-amber: #fbbf24;
            --accent-orange: #fb923c;
            --accent-rose: #f43f5e;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --text-muted: #64748b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }

        /* Mode Selector Tab in Sidebar */
        .mode-tab-group { display: flex; padding: 12px; gap: 6px; background: rgba(0,0,0,0.3); border-bottom: 1px solid var(--card-border); }
        .mode-tab { flex: 1; padding: 9px 10px; border: 1px solid var(--card-border); background: var(--card-sub-bg); color: var(--text-sub); border-radius: 8px; font-size: 0.76rem; font-weight: 700; cursor: pointer; text-align: center; transition: all 0.15s ease; }
        .mode-tab:hover { border-color: var(--accent-cyan); color: var(--text-main); }
        .mode-tab.active { background: linear-gradient(135deg, #0284c7, #0369a1); color: #fff; border-color: #38bdf8; box-shadow: 0 2px 10px rgba(2, 132, 199, 0.4); }

        /* Sidebar */
        .sidebar { width: 290px; min-width: 290px; background-color: var(--sidebar-bg); border-right: 1px solid var(--card-border); display: flex; flex-direction: column; }
        .sidebar-header { padding: 18px 20px; border-bottom: 1px solid var(--card-border); }
        .sidebar-header h1 { font-size: 1.15rem; font-weight: 900; color: var(--accent-cyan); display: flex; align-items: center; gap: 8px; letter-spacing: -0.02em; }
        .sidebar-header p { font-size: 0.74rem; color: var(--text-sub); margin-top: 4px; }
        
        .section-label { padding: 14px 18px 6px; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.08em; display: flex; justify-content: space-between; align-items: center; }
        .nav-list { list-style: none; overflow-y: auto; flex: 1; padding: 4px 10px 16px; }
        .nav-item { margin-bottom: 4px; }
        .nav-btn { width: 100%; text-align: left; padding: 10px 12px; background: transparent; border: 1px solid transparent; color: var(--text-sub); border-radius: 8px; cursor: pointer; font-size: 0.83rem; font-weight: 600; display: flex; justify-content: space-between; align-items: center; transition: all 0.15s ease; }
        .nav-btn:hover { background-color: rgba(56, 189, 248, 0.08); color: var(--text-main); border-color: rgba(56, 189, 248, 0.2); }
        .nav-btn.active { background: linear-gradient(135deg, #0284c7, #0369a1); color: #fff; border-color: #38bdf8; font-weight: 700; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3); }

        /* Main Container */
        .main-container { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
        
        /* Top Navigation Bar */
        .top-bar { padding: 14px 24px; background-color: var(--card-bg); border-bottom: 1px solid var(--card-border); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; z-index: 10; }
        .header-info { display: flex; align-items: baseline; gap: 12px; }
        .view-title { font-size: 1.35rem; font-weight: 800; color: var(--text-main); letter-spacing: -0.02em; }
        .badge-pill { font-size: 0.82rem; color: var(--accent-cyan); font-weight: 700; font-family: 'JetBrains Mono', monospace; background: rgba(56, 189, 248, 0.12); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.25); }
        
        .view-controls { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
        .filter-btn { padding: 6px 12px; background: rgba(255,255,255,0.04); border: 1px solid var(--card-border); color: var(--text-sub); border-radius: 6px; font-size: 0.75rem; font-weight: 600; cursor: pointer; transition: all 0.15s ease; white-space: nowrap; }
        .filter-btn:hover { border-color: var(--accent-cyan); color: var(--text-main); background: rgba(56, 189, 248, 0.08); }
        .filter-btn.active { background: var(--accent-cyan); color: #000; border-color: var(--accent-cyan); font-weight: 700; }

        .search-input { padding: 6px 12px; background: #0c1220; border: 1px solid var(--card-border); color: var(--text-main); border-radius: 6px; font-size: 0.78rem; outline: none; width: 220px; font-family: 'Inter', sans-serif; }
        .search-input:focus { border-color: var(--accent-cyan); box-shadow: 0 0 8px rgba(56, 189, 248, 0.3); }

        /* Scroll Area */
        .content-scroll { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 20px; }

        /* Boxes & Cards */
        .box { background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 18px 20px; display: flex; flex-direction: column; gap: 14px; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }
        .box-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
        .box-title { font-size: 0.96rem; font-weight: 700; color: var(--text-main); display: flex; align-items: center; gap: 8px; }
        .badge-count { font-size: 0.74rem; background: rgba(56, 189, 248, 0.15); color: var(--accent-cyan); padding: 2px 8px; border-radius: 10px; font-weight: 700; }
        
        /* Scores Table */
        .score-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; text-align: left; margin-top: 4px; }
        .score-table th { background: #0c1220; color: var(--text-sub); padding: 10px 12px; font-weight: 700; border: 1px solid var(--card-border); }
        .score-table td { padding: 9px 12px; border: 1px solid var(--card-border); color: var(--text-main); }
        .score-table tr:hover td { background: rgba(56, 189, 248, 0.05); }
        .score-tag { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; }

        /* Horizontal Thumb Grid */
        .horizon-grid { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px; scrollbar-width: thin; }
        .thumb-card { width: 140px; flex-shrink: 0; background: var(--card-sub-bg); border: 1px solid var(--card-border); border-radius: 8px; overflow: hidden; cursor: pointer; transition: all 0.2s ease; }
        .thumb-card:hover { transform: translateY(-3px); border-color: var(--accent-cyan); box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25); }
        .thumb-img { width: 140px; height: 140px; object-fit: cover; background: #000; display: block; }
        .thumb-info { padding: 6px 8px; font-size: 0.72rem; color: var(--text-sub); text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-family: monospace; }

        /* Concept Matrix Grid */
        .prompt-matrix { display: flex; flex-direction: column; gap: 16px; }
        .prompt-row { background: var(--card-sub-bg); border: 1px solid var(--card-border); border-radius: 10px; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
        .prompt-row-header { display: flex; align-items: center; gap: 10px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 10px; }
        .prompt-idx-badge { background: var(--accent-cyan); color: #000; font-weight: 800; font-size: 0.75rem; padding: 3px 8px; border-radius: 4px; font-family: monospace; }
        .prompt-text { font-size: 0.88rem; font-weight: 600; color: var(--text-main); font-family: 'JetBrains Mono', monospace; line-height: 1.4; }

        .exp-comparison-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
        .exp-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; transition: all 0.2s ease; cursor: pointer; }
        .exp-card:hover { border-color: var(--accent-cyan); transform: translateY(-3px); box-shadow: 0 6px 18px rgba(0,0,0,0.5); }
        .exp-card-header { padding: 6px 10px; background: rgba(0,0,0,0.4); font-size: 0.74rem; font-weight: 700; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.04); }
        .exp-img-wrapper { width: 100%; aspect-ratio: 1/1; background: #000; overflow: hidden; position: relative; }
        .exp-img-wrapper img { width: 100%; height: 100%; object-fit: contain; display: block; }

        /* ================================================================= */
        /* MODE 2: Large 100-Image Gallery Grid Layout                        */
        /* ================================================================= */
        .gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 18px; }
        .gallery-card { background: var(--card-sub-bg); border: 1px solid var(--card-border); border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; transition: all 0.22s ease; cursor: pointer; box-shadow: 0 4px 16px rgba(0,0,0,0.3); }
        .gallery-card:hover { border-color: var(--accent-cyan); transform: translateY(-4px); box-shadow: 0 8px 24px rgba(56, 189, 248, 0.25); }
        .gallery-img-box { width: 100%; aspect-ratio: 1/1; background: #000; overflow: hidden; position: relative; }
        .gallery-img-box img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform 0.25s ease; }
        .gallery-card:hover .gallery-img-box img { transform: scale(1.03); }
        
        .gallery-card-body { padding: 14px 16px; display: flex; flex-direction: column; gap: 8px; flex: 1; }
        .gallery-card-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
        .gallery-concept-tag { font-size: 0.74rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; padding: 2px 8px; border-radius: 4px; background: rgba(56, 189, 248, 0.15); color: var(--accent-cyan); }
        .gallery-idx-tag { font-size: 0.72rem; color: var(--text-sub); font-family: monospace; font-weight: 600; }
        
        .gallery-prompt-text { font-size: 0.82rem; color: var(--text-main); font-weight: 500; font-family: 'JetBrains Mono', monospace; line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
        
        .gallery-score-row { display: flex; justify-content: space-between; align-items: center; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 0.74rem; }
        .score-pill { padding: 2px 6px; border-radius: 4px; font-weight: 700; font-family: monospace; }
        .score-pill.t2i { background: rgba(56, 189, 248, 0.15); color: var(--accent-cyan); }
        .score-pill.i2i { background: rgba(74, 222, 128, 0.15); color: var(--accent-green); }
        .score-pill.total { background: rgba(192, 132, 252, 0.18); color: var(--accent-purple); }

        /* Modal */
        .modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.92); backdrop-filter: blur(8px); z-index: 1000; align-items: center; justify-content: center; padding: 20px; }
        .modal.active { display: flex; }
        .modal-content { max-width: 92vw; max-height: 94vh; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 14px; overflow: hidden; display: flex; flex-direction: column; box-shadow: 0 16px 50px rgba(0,0,0,0.8); }
        .modal-body { padding: 16px; background: #000; flex: 1; display: flex; align-items: center; justify-content: center; min-height: 400px; }
        .modal-body img { max-width: 100%; max-height: 75vh; object-fit: contain; border-radius: 6px; }
        .modal-footer { padding: 14px 22px; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--card-border); background: var(--card-sub-bg); flex-wrap: wrap; gap: 10px; }
        .close-btn { background: var(--accent-cyan); color: #000; border: none; padding: 8px 18px; border-radius: 6px; font-weight: 800; cursor: pointer; transition: all 0.15s; }
        .close-btn:hover { background: #0284c7; color: #fff; }
    </style>
</head>
<body>
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-header">
            <h1>🔬 VERILUX Viewer</h1>
            <p>Multi-Subject Generation Dashboard</p>
        </div>
        
        <!-- View Mode Switcher -->
        <div class="mode-tab-group">
            <button class="mode-tab active" id="tabExpMode" onclick="switchViewMode('experiment')">🚀 Experiment 100-View</button>
            <button class="mode-tab" id="tabConceptMode" onclick="switchViewMode('concept')">📌 Concept Matrix</button>
        </div>

        <div class="section-label" id="sidebarSectionLabel">
            <span>EXPERIMENTS (12개)</span>
        </div>
        <ul class="nav-list" id="sidebarNav"></ul>
    </div>

    <!-- Main Container -->
    <div class="main-container">
        <!-- Top Bar -->
        <div class="top-bar">
            <div class="header-info">
                <span class="view-title" id="mainTitle">Loading...</span>
                <span class="badge-pill" id="mainSubBadge"></span>
            </div>
            
            <div class="view-controls" id="topControls">
                <!-- Dynamically populated based on mode -->
            </div>
        </div>

        <!-- Scroll Content -->
        <div class="content-scroll" id="contentArea">
            <!-- Dynamically populated -->
        </div>
    </div>

    <!-- Modal -->
    <div class="modal" id="modal" onclick="closeModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-body">
                <img id="modalImg" src="" alt="View">
            </div>
            <div class="modal-footer">
                <div style="flex:1;">
                    <div id="modalTitle" style="font-weight:800; color:var(--text-main); font-size:1.05rem;"></div>
                    <div id="modalMeta" style="font-size:0.84rem; color:var(--accent-cyan); font-family:'JetBrains Mono', monospace; margin-top:4px; line-height:1.4;"></div>
                </div>
                <button class="close-btn" onclick="closeModal()">Close (ESC)</button>
            </div>
        </div>
    </div>

    <script>
        const DATA = __DATA_JSON__;
        let currentMode = 'experiment'; // 'experiment' default
        let currentConcept = Object.keys(DATA.concepts)[0];
        let currentExp = DATA.experiments[DATA.experiments.length - 1] || '13_sota_ensemble';
        let selectedExps = [...DATA.experiments];
        let galleryConceptFilter = 'ALL';
        let gallerySearchQuery = '';

        function init() {
            switchViewMode('experiment');
        }

        function switchViewMode(mode) {
            currentMode = mode;
            document.getElementById('tabExpMode').className = `mode-tab ${mode === 'experiment' ? 'active' : ''}`;
            document.getElementById('tabConceptMode').className = `mode-tab ${mode === 'concept' ? 'active' : ''}`;
            
            if (mode === 'experiment') {
                document.getElementById('sidebarSectionLabel').innerHTML = '<span>EXPERIMENTS (' + DATA.experiments.length + '개)</span>';
                renderExperimentSidebar();
                selectExperiment(currentExp);
            } else {
                document.getElementById('sidebarSectionLabel').innerHTML = '<span>CONCEPTS (10개)</span>';
                renderConceptSidebar();
                selectConcept(currentConcept);
            }
        }

        /* ================================================================= */
        /* MODE 2: EXPERIMENT SIDEBAR & 100-IMAGE GALLERY VIEW                */
        /* ================================================================= */
        function renderExperimentSidebar() {
            const nav = document.getElementById('sidebarNav');
            nav.innerHTML = '';
            
            DATA.experiments.forEach(exp => {
                const meta = DATA.exp_meta[exp] || { name: exp, color: '#94a3b8', tag: '-' };
                const li = document.createElement('li');
                li.className = 'nav-item';
                
                const btn = document.createElement('button');
                btn.className = `nav-btn ${exp === currentExp ? 'active' : ''}`;
                btn.onclick = () => selectExperiment(exp);
                btn.innerHTML = `
                    <div style="display:flex; flex-direction:column; gap:2px; text-align:left;">
                        <span style="font-weight:700;">${exp}</span>
                        <span style="font-size:0.7rem; opacity:0.8; color:${meta.color};">${meta.name}</span>
                    </div>
                `;
                li.appendChild(btn);
                nav.appendChild(li);
            });
        }

        function selectExperiment(exp) {
            currentExp = exp;
            renderExperimentSidebar();
            
            const meta = DATA.exp_meta[exp] || { name: exp, color: '#38bdf8', tag: '-' };
            document.getElementById('mainTitle').innerText = meta.name;
            document.getElementById('mainSubBadge').innerText = `100-Image Matrix (${meta.tag})`;
            
            renderExperimentTopControls();
            renderGalleryContent();
        }

        function renderExperimentTopControls() {
            const group = document.getElementById('topControls');
            group.innerHTML = '';
            
            const allBtn = document.createElement('button');
            allBtn.className = `filter-btn ${galleryConceptFilter === 'ALL' ? 'active' : ''}`;
            allBtn.innerText = '전체 서브젝트 (100장)';
            allBtn.onclick = () => {
                galleryConceptFilter = 'ALL';
                renderExperimentTopControls();
                renderGalleryContent();
            };
            group.appendChild(allBtn);

            Object.keys(DATA.concepts).forEach(concept => {
                const btn = document.createElement('button');
                btn.className = `filter-btn ${galleryConceptFilter === concept ? 'active' : ''}`;
                btn.innerText = concept;
                btn.onclick = () => {
                    galleryConceptFilter = concept;
                    renderExperimentTopControls();
                    renderGalleryContent();
                };
                group.appendChild(btn);
            });

            const search = document.createElement('input');
            search.type = 'text';
            search.className = 'search-input';
            search.placeholder = '🔍 프롬프트 실시간 검색...';
            search.value = gallerySearchQuery;
            search.oninput = (e) => {
                gallerySearchQuery = e.target.value.toLowerCase();
                renderGalleryContent();
            };
            group.appendChild(search);
        }

        function renderGalleryContent() {
            const area = document.getElementById('contentArea');
            const exp = currentExp;
            const meta = DATA.exp_meta[exp] || { name: exp, color: '#38bdf8' };

            let avgT = '-', avgI = '-', avgTot = '-';
            if (DATA.scores[exp] && DATA.scores[exp].average_scores) {
                const av = DATA.scores[exp].average_scores;
                avgT = (av['CLIP-T'] || av.t2i || 0).toFixed(4);
                avgI = (av['CLIP-I'] || av.i2i || 0).toFixed(4);
                avgTot = (av['CLIP-Total'] || av.total || (av.t2i + av.i2i) || 0).toFixed(4);
            }

            let cardsHtml = '';
            let count = 0;

            Object.keys(DATA.concepts).forEach(concept => {
                if (galleryConceptFilter !== 'ALL' && galleryConceptFilter !== concept) return;

                const info = DATA.concepts[concept];
                const prompts = info.prompts;
                const expImgs = info.exp_generated[exp] || {};
                const selMetaList = (DATA.selection_meta[exp] && DATA.selection_meta[exp][concept]) || [];

                prompts.forEach((pText, pIdx) => {
                    if (gallerySearchQuery && !pText.toLowerCase().includes(gallerySearchQuery) && !concept.toLowerCase().includes(gallerySearchQuery)) {
                        return;
                    }

                    const imgInfo = expImgs[pIdx];
                    if (!imgInfo) return;

                    count++;
                    let t2iStr = '-', i2iStr = '-', totalStr = '-';
                    const selObj = selMetaList.find(s => s.prompt_idx === pIdx);
                    if (selObj) {
                        t2iStr = selObj.clip_t.toFixed(4);
                        i2iStr = selObj.clip_i.toFixed(4);
                        totalStr = (selObj.clip_t + selObj.clip_i).toFixed(4);
                    } else if (DATA.scores[exp] && DATA.scores[exp].per_concept_scores && DATA.scores[exp].per_concept_scores[concept]) {
                        const cs = DATA.scores[exp].per_concept_scores[concept];
                        t2iStr = cs.t2i ? cs.t2i.toFixed(4) : '-';
                        i2iStr = cs.i2i ? cs.i2i.toFixed(4) : '-';
                        if (cs.t2i && cs.i2i) totalStr = (cs.t2i + cs.i2i).toFixed(4);
                    }

                    cardsHtml += `
                        <div class="gallery-card" onclick="openModal('${imgInfo.rel_path}', '${meta.name} - [${concept}]', 'Prompt #${pIdx}: \\'${pText}\\' | T: ${t2iStr}, I: ${i2iStr}')">
                            <div class="gallery-img-box">
                                <img src="${imgInfo.rel_path}" loading="lazy" alt="${concept}">
                            </div>
                            <div class="gallery-card-body">
                                <div class="gallery-card-top">
                                    <span class="gallery-concept-tag">${concept}</span>
                                    <span class="gallery-idx-tag">#${pIdx}</span>
                                </div>
                                <div class="gallery-prompt-text" title="${pText}">
                                    "${pText}"
                                </div>
                                <div class="gallery-score-row">
                                    <span class="score-pill t2i">T: ${t2iStr}</span>
                                    <span class="score-pill i2i">I: ${i2iStr}</span>
                                    <span class="score-pill total">Total: ${totalStr}</span>
                                </div>
                            </div>
                        </div>
                    `;
                });
            });

            area.innerHTML = `
                <!-- Summary Banner -->
                <div class="box" style="background:linear-gradient(135deg, rgba(2, 132, 199, 0.12), rgba(15, 23, 42, 0.6)); border-color:rgba(56, 189, 248, 0.25);">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                        <div>
                            <div style="font-size:1.15rem; font-weight:800; color:var(--text-main);">${meta.name}</div>
                            <div style="font-size:0.8rem; color:var(--text-sub); margin-top:2px;">알고리즘: <b>${meta.tag}</b></div>
                        </div>
                        <div style="display:flex; gap:10px; align-items:center;">
                            <div class="score-pill t2i" style="font-size:0.8rem; padding:4px 10px;">공식 CLIP-T: ${avgT}</div>
                            <div class="score-pill i2i" style="font-size:0.8rem; padding:4px 10px;">공식 CLIP-I: ${avgI}</div>
                            <div class="score-pill total" style="font-size:0.8rem; padding:4px 10px;">종합 Total: ${avgTot}</div>
                        </div>
                    </div>
                </div>

                <!-- 100-Image Gallery -->
                <div class="box">
                    <div class="box-header">
                        <div class="box-title">
                            <span>🖼️ ${meta.name} 생성 이미지 갤러리</span>
                            <span class="badge-count">${count}장 표시 중</span>
                        </div>
                    </div>
                    <div class="gallery-grid">
                        ${cardsHtml || '<div style="color:var(--text-sub); padding:30px;">검색 조건에 맞는 이미지가 없습니다.</div>'}
                    </div>
                </div>
            `;
        }

        /* ================================================================= */
        /* MODE 1: CONCEPT MATRIX VIEW                                       */
        /* ================================================================= */
        function renderConceptSidebar() {
            const nav = document.getElementById('sidebarNav');
            nav.innerHTML = '';
            
            Object.keys(DATA.concepts).forEach(concept => {
                const item = DATA.concepts[concept];
                const li = document.createElement('li');
                li.className = 'nav-item';
                
                const btn = document.createElement('button');
                btn.className = `nav-btn ${concept === currentConcept ? 'active' : ''}`;
                btn.onclick = () => selectConcept(concept);
                btn.innerHTML = `
                    <span>${concept}</span>
                    <span style="font-size:0.72rem; opacity:0.8;">${item.orig_images.length}장</span>
                `;
                li.appendChild(btn);
                nav.appendChild(li);
            });
        }

        function selectConcept(concept) {
            currentConcept = concept;
            renderConceptSidebar();
            
            const info = DATA.concepts[concept];
            document.getElementById('mainTitle').innerText = concept;
            document.getElementById('mainSubBadge').innerText = `class: "${info.class_prompt}"`;
            
            renderConceptTopControls();
            renderConceptContent(concept, info);
        }

        function renderConceptTopControls() {
            const group = document.getElementById('topControls');
            group.innerHTML = '<span style="font-size:0.75rem; color:var(--text-muted); margin-right:4px;">Filter Exps:</span>';
            
            const allBtn = document.createElement('button');
            allBtn.className = `filter-btn ${selectedExps.length === DATA.experiments.length ? 'active' : ''}`;
            allBtn.innerText = '전체 (ALL)';
            allBtn.onclick = () => {
                selectedExps = [...DATA.experiments];
                renderConceptTopControls();
                renderConceptMatrix();
            };
            group.appendChild(allBtn);

            DATA.experiments.forEach(exp => {
                const btn = document.createElement('button');
                const isSelected = selectedExps.includes(exp);
                btn.className = `filter-btn ${isSelected && selectedExps.length !== DATA.experiments.length ? 'active' : ''}`;
                btn.innerText = exp.replace(/^[0-9]+_/, '');
                btn.title = exp;
                btn.onclick = () => {
                    if (selectedExps.includes(exp)) {
                        if (selectedExps.length > 1) selectedExps = selectedExps.filter(e => e !== exp);
                    } else {
                        selectedExps.push(exp);
                    }
                    renderConceptTopControls();
                    renderConceptMatrix();
                };
                group.appendChild(btn);
            });
        }

        function renderConceptContent(concept, info) {
            const area = document.getElementById('contentArea');
            area.innerHTML = `
                <!-- Scores Table -->
                <div class="box">
                    <div class="box-header">
                        <div class="box-title">
                            <span>📊 서브젝트별 정량 평가 지표 (Quantitative Benchmark Scores)</span>
                        </div>
                    </div>
                    <div style="overflow-x: auto;">
                        <table class="score-table">
                            <thead>
                                <tr>
                                    <th>실험 ID</th>
                                    <th>실험명 & 핵심 알고리즘</th>
                                    <th>공식 CLIP-T (↑)</th>
                                    <th>공식 CLIP-I (↑)</th>
                                    <th>Total (T+I)</th>
                                    <th>DINO-I (↑)</th>
                                    <th>Diversity</th>
                                </tr>
                            </thead>
                            <tbody id="conceptScoreBody"></tbody>
                        </table>
                    </div>
                </div>

                <!-- References -->
                <div class="box">
                    <div class="box-header">
                        <div class="box-title">
                            <span>📸 원본(Dataset) 및 전처리 증강(Augmentation) 참조 이미지</span>
                        </div>
                    </div>
                    <div style="display:flex; flex-direction:column; gap:14px;">
                        <div>
                            <div style="font-size:0.8rem; font-weight:700; color:var(--text-sub); margin-bottom:8px;">• 원본 이미지 (dataset/) <span id="origCount" class="badge-count">${info.orig_images.length}장</span></div>
                            <div class="horizon-grid" id="origGrid"></div>
                        </div>
                        <div>
                            <div style="font-size:0.8rem; font-weight:700; color:var(--text-sub); margin-bottom:8px;">• 전처리 증강 이미지 (augmentation/) <span id="augCount" class="badge-count">${info.aug_images.length}장</span></div>
                            <div class="horizon-grid" id="augGrid"></div>
                        </div>
                    </div>
                </div>

                <!-- Prompt Matrix -->
                <div class="box">
                    <div class="box-header">
                        <div class="box-title">
                            <span>🎯 10개 프롬프트별 실험 생성 이미지 비교 Matrix</span>
                        </div>
                    </div>
                    <div class="prompt-matrix" id="promptMatrix"></div>
                </div>
            `;

            renderConceptScores(concept);
            renderReferenceImages(info);
            renderConceptMatrix();
        }

        function renderConceptScores(concept) {
            const tbody = document.getElementById('conceptScoreBody');
            tbody.innerHTML = '';

            DATA.experiments.forEach(exp => {
                const tr = document.createElement('tr');
                const meta = DATA.exp_meta[exp] || { name: exp, color: '#94a3b8', tag: '-' };
                
                let t2i = '-', i2i = '-', total = '-', dino = '-', div = '-';

                if (DATA.scores[exp] && DATA.scores[exp].per_concept_scores && DATA.scores[exp].per_concept_scores[concept]) {
                    const s = DATA.scores[exp].per_concept_scores[concept];
                    t2i = s.t2i !== undefined ? s.t2i.toFixed(4) : '-';
                    i2i = s.i2i !== undefined ? s.i2i.toFixed(4) : '-';
                    if (s.t2i !== undefined && s.i2i !== undefined) {
                        total = (s.t2i + s.i2i).toFixed(4);
                    }
                }

                if (DATA.extended_scores[exp] && DATA.extended_scores[exp].per_concept && DATA.extended_scores[exp].per_concept[concept]) {
                    const es = DATA.extended_scores[exp].per_concept[concept];
                    dino = es.dino_i !== undefined ? es.dino_i.toFixed(4) : '-';
                    div = es.diversity !== undefined ? es.diversity.toFixed(4) : '-';
                }

                tr.innerHTML = `
                    <td><span class="score-tag" style="background:rgba(255,255,255,0.06); color:${meta.color};">${exp}</span></td>
                    <td><b>${meta.name}</b> <span style="font-size:0.72rem; color:var(--text-sub);">(${meta.tag})</span></td>
                    <td style="color:var(--accent-cyan); font-weight:700;">${t2i}</td>
                    <td style="color:var(--accent-green); font-weight:700;">${i2i}</td>
                    <td style="color:#fff; font-weight:800;">${total}</td>
                    <td style="color:var(--accent-purple); font-weight:600;">${dino}</td>
                    <td style="color:var(--text-sub);">${div}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function renderReferenceImages(info) {
            const origGrid = document.getElementById('origGrid');
            origGrid.innerHTML = info.orig_images.length === 0 ? '<span style="font-size:0.8rem; color:var(--text-sub);">없음</span>' : '';
            info.orig_images.forEach(img => {
                const card = document.createElement('div');
                card.className = 'thumb-card';
                card.onclick = () => openModal(img.rel_path, `원본 이미지: ${currentConcept}`, `파일명: ${img.filename} | 해상도: ${img.width}x${img.height} | 용량: ${img.size_kb} KB`);
                card.innerHTML = `
                    <img class="thumb-img" src="${img.rel_path}" loading="lazy">
                    <div class="thumb-info" title="${img.filename}">${img.filename}</div>
                `;
                origGrid.appendChild(card);
            });

            const augGrid = document.getElementById('augGrid');
            augGrid.innerHTML = info.aug_images.length === 0 ? '<span style="font-size:0.8rem; color:var(--text-sub);">증강 데이터 없음</span>' : '';
            info.aug_images.forEach(img => {
                const card = document.createElement('div');
                card.className = 'thumb-card';
                card.onclick = () => openModal(img.rel_path, `증강 이미지: ${currentConcept}`, `파일명: ${img.filename} | 해상도: ${img.width}x${img.height} | 용량: ${img.size_kb} KB`);
                card.innerHTML = `
                    <img class="thumb-img" src="${img.rel_path}" loading="lazy">
                    <div class="thumb-info" title="${img.filename}">${img.filename}</div>
                `;
                augGrid.appendChild(card);
            });
        }

        function renderConceptMatrix() {
            const matrix = document.getElementById('promptMatrix');
            if (!matrix) return;
            matrix.innerHTML = '';
            
            const info = DATA.concepts[currentConcept];
            const prompts = info.prompts;

            if (!prompts || prompts.length === 0) {
                matrix.innerHTML = '<div style="color:var(--text-sub); padding:20px;">프롬프트 정보가 없습니다.</div>';
                return;
            }

            prompts.forEach((pText, pIdx) => {
                const row = document.createElement('div');
                row.className = 'prompt-row';
                
                const rowHeader = document.createElement('div');
                rowHeader.className = 'prompt-row-header';
                rowHeader.innerHTML = `
                    <span class="prompt-idx-badge">#${pIdx}</span>
                    <span class="prompt-text">${pText}</span>
                `;
                row.appendChild(rowHeader);

                const grid = document.createElement('div');
                grid.className = 'exp-comparison-grid';

                selectedExps.forEach(exp => {
                    const expImgs = info.exp_generated[exp] || {};
                    const imgInfo = expImgs[pIdx];
                    const meta = DATA.exp_meta[exp] || { name: exp, color: '#38bdf8' };

                    const card = document.createElement('div');
                    card.className = 'exp-card';

                    if (imgInfo) {
                        card.onclick = () => openModal(imgInfo.rel_path, `${meta.name} - Prompt #${pIdx}`, `[${currentConcept}] "${pText}"`);
                        card.innerHTML = `
                            <div class="exp-card-header">
                                <span style="color:${meta.color};">${exp.replace(/^[0-9]+_/, '')}</span>
                                <span style="color:var(--text-muted);">#${pIdx}</span>
                            </div>
                            <div class="exp-img-wrapper">
                                <img src="${imgInfo.rel_path}" loading="lazy">
                            </div>
                        `;
                    } else {
                        card.innerHTML = `
                            <div class="exp-card-header">
                                <span>${exp.replace(/^[0-9]+_/, '')}</span>
                                <span>-</span>
                            </div>
                            <div class="exp-img-wrapper" style="display:flex; align-items:center; justify-content:center; color:var(--text-muted); font-size:0.75rem;">
                                N/A
                            </div>
                        `;
                    }
                    grid.appendChild(card);
                });

                row.appendChild(grid);
                matrix.appendChild(row);
            });
        }

        function openModal(src, title, meta) {
            document.getElementById('modalImg').src = src;
            document.getElementById('modalTitle').innerText = title;
            document.getElementById('modalMeta').innerText = meta;
            document.getElementById('modal').classList.add('active');
        }

        function closeModal(e) {
            document.getElementById('modal').classList.remove('active');
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });

        init();
    </script>
</body>
</html>
"""


def generate_html(data, output_html_path):
    json_data = json.dumps(data, ensure_ascii=False)
    html_content = HTML_TEMPLATE.replace("__DATA_JSON__", json_data)

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✓ Dual-View Experiment Viewer HTML이 성공적으로 생성되었습니다: {output_html_path}")


def start_server(html_path, host="0.0.0.0", port=8500):
    root_dir = os.path.dirname(os.path.abspath(html_path))
    os.chdir(root_dir)
    
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer((host, port), handler)
    
    file_name = os.path.basename(html_path)
    local_url = f"http://localhost:{port}/{file_name}"
    
    print(f"\n🚀 대시보드 웹 서버 구동 중:")
    print(f"  • 접속 URL: {local_url}")
    print("종료하려면 Ctrl+C를 누르세요.\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n서버가 종료되었습니다.")
        httpd.server_close()


def main():
    default_root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Experiment & Generation Dashboard Generator")
    parser.add_argument("--root", type=str, default=default_root, help="프로젝트 루트 폴더")
    parser.add_argument("--out", type=str, default="experiment_viewer.html", help="생성할 HTML 파일명")
    parser.add_argument("--server", action="store_true", help="생성 후 HTTP 서버 자동 구동")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="바인딩 호스트 주소")
    parser.add_argument("--port", type=int, default=8500, help="웹 서버 포트 번호")

    args = parser.parse_args()
    
    print("프로젝트 데이터 및 실험 결과 스캔 중...")
    data = scan_all(args.root)
    out_path = os.path.join(args.root, args.out)
    generate_html(data, out_path)
    
    if args.server:
        start_server(out_path, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
