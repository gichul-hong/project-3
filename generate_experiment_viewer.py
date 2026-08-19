"""
Experiment & Image Comparison Dashboard Generator
-------------------------------------------------
dataset/, augmentation/, prompt/, experiments/ 폴더를 자동으로 스캔하여
원본 이미지, 증강 이미지, Iteration(실험)별 Prompt 기반 생성 이미지를
하나의 뷰(Interactive Web Dashboard)에서 비교·조망할 수 있는
experiment_viewer.html 을 생성합니다.

사용법:
    python generate_experiment_viewer.py [--server] [--port 8500]
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
            if os.path.isdir(os.path.join(experiments_dir, f))
        ])

    data = {
        "concepts": {},
        "experiments": exp_folders,
        "scores": {}
    }

    # 실험별 평가 점수 (eval_summary.json) 스캔
    for exp in exp_folders:
        summary_path = os.path.join(experiments_dir, exp, "eval_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data["scores"][exp] = json.load(f)
            except Exception:
                pass

    # 컨셉별 이미지 및 프롬프트 스캔
    for concept, class_prompt in CLASS_PROMPT.items():
        # 1. 원본 이미지
        orig_dir = os.path.join(root_dir, "dataset", concept)
        orig_imgs = []
        if os.path.exists(orig_dir):
            paths = sorted(
                glob.glob(os.path.join(orig_dir, "*.png")) +
                glob.glob(os.path.join(orig_dir, "*.jpg")) +
                glob.glob(os.path.join(orig_dir, "*.jpeg"))
            )
            orig_imgs = [get_image_info(p, root_dir) for p in paths]

        # 2. 증강 이미지
        aug_dir = os.path.join(root_dir, "augmentation", concept)
        aug_imgs = []
        if os.path.exists(aug_dir):
            paths = sorted(
                glob.glob(os.path.join(aug_dir, "*.png")) +
                glob.glob(os.path.join(aug_dir, "*.jpg")) +
                glob.glob(os.path.join(aug_dir, "*.jpeg"))
            )
            aug_imgs = [get_image_info(p, root_dir) for p in paths]

        # 3. 테스트 프롬프트
        prompt_file = os.path.join(root_dir, "prompt", f"{concept}.txt")
        prompts = []
        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompts = [
                    l.strip().replace("{}", class_prompt)
                    for l in f.readlines() if l.strip()
                ]

        # 4. Iteration(실험)별 생성 이미지
        exp_generated = {}
        for exp in exp_folders:
            concept_exp_dir = os.path.join(experiments_dir, exp, concept)
            exp_imgs = {}
            if os.path.exists(concept_exp_dir):
                for idx in range(len(prompts) if prompts else 10):
                    # 0.png, 1.png ... 또는 파일 순서
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


def generate_html(data, output_html_path):
    json_data = json.dumps(data, ensure_ascii=False)
    
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment & Generation Dashboard</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151d30;
            --card-border: #232f48;
            --accent-color: #38bdf8;
            --accent-hover: #0284c7;
            --accent-green: #4ade80;
            --accent-purple: #c084fc;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        body {{ background-color: var(--bg-color); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }}

        /* Sidebar */
        .sidebar {{ width: 280px; background-color: #070a12; border-right: 1px solid var(--card-border); display: flex; flex-direction: column; }}
        .sidebar-header {{ padding: 20px; border-bottom: 1px solid var(--card-border); }}
        .sidebar-header h1 {{ font-size: 1.15rem; font-weight: 800; color: var(--accent-color); display: flex; align-items: center; gap: 8px; }}
        .sidebar-header p {{ font-size: 0.78rem; color: var(--text-sub); margin-top: 4px; }}
        
        .section-label {{ padding: 12px 16px 4px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: var(--text-sub); letter-spacing: 0.05em; }}
        .nav-list {{ list-style: none; overflow-y: auto; flex: 1; padding: 4px 8px 16px; }}
        .nav-item {{ margin-bottom: 3px; }}
        .nav-btn {{ width: 100%; text-align: left; padding: 10px 12px; background: transparent; border: none; color: var(--text-sub); border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 600; display: flex; justify-content: space-between; align-items: center; transition: all 0.15s; }}
        .nav-btn:hover {{ background-color: rgba(56, 189, 248, 0.08); color: var(--text-main); }}
        .nav-btn.active {{ background-color: var(--accent-color); color: #000; font-weight: 700; }}

        /* Main Container */
        .main-container {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
        
        /* Top Navigation Bar */
        .top-bar {{ padding: 14px 24px; background-color: var(--card-bg); border-bottom: 1px solid var(--card-border); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }}
        .concept-info {{ display: flex; align-items: baseline; gap: 10px; }}
        .concept-title {{ font-size: 1.35rem; font-weight: 800; color: var(--text-main); }}
        .concept-class {{ font-size: 0.88rem; color: var(--accent-color); font-weight: 600; }}
        
        .view-controls {{ display: flex; gap: 8px; align-items: center; }}
        .filter-btn {{ padding: 6px 12px; background: rgba(255,255,255,0.05); border: 1px solid var(--card-border); color: var(--text-sub); border-radius: 6px; font-size: 0.8rem; font-weight: 600; cursor: pointer; transition: all 0.2s; }}
        .filter-btn:hover {{ border-color: var(--accent-color); color: var(--text-main); }}
        .filter-btn.active {{ background: var(--accent-color); color: #000; border-color: var(--accent-color); font-weight: 700; }}

        /* Scroll Area */
        .content-scroll {{ flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 24px; }}

        /* Collapsible Section Box */
        .box {{ background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 18px 20px; display: flex; flex-direction: column; gap: 14px; }}
        .box-header {{ display: flex; justify-content: space-between; align-items: center; }}
        .box-title {{ font-size: 1rem; font-weight: 700; color: var(--text-main); display: flex; align-items: center; gap: 8px; }}
        .badge-count {{ font-size: 0.75rem; background: rgba(56, 189, 248, 0.15); color: var(--accent-color); padding: 2px 8px; border-radius: 10px; font-weight: 600; }}
        
        /* Grid Layouts */
        .horizon-grid {{ display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px; }}
        .thumb-card {{ width: 140px; flex-shrink: 0; background: #0c121e; border: 1px solid var(--card-border); border-radius: 8px; overflow: hidden; cursor: pointer; transition: transform 0.2s; }}
        .thumb-card:hover {{ transform: translateY(-3px); border-color: var(--accent-color); }}
        .thumb-img {{ width: 140px; height: 140px; object-fit: cover; background: #000; }}
        .thumb-info {{ padding: 6px 8px; font-size: 0.72rem; color: var(--text-sub); text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

        /* Comparison Table / Matrix */
        .prompt-matrix {{ display: flex; flex-direction: column; gap: 16px; }}
        .prompt-row {{ background: #0e1626; border: 1px solid var(--card-border); border-radius: 10px; padding: 14px 16px; display: flex; flex-direction: column; gap: 12px; }}
        .prompt-row-header {{ display: flex; align-items: center; gap: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px; }}
        .prompt-idx-badge {{ background: var(--accent-color); color: #000; font-weight: 800; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; }}
        .prompt-text {{ font-size: 0.9rem; font-weight: 600; color: var(--text-main); font-family: monospace; }}

        .exp-comparison-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 14px; }}
        .exp-card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; transition: border-color 0.2s; cursor: pointer; }}
        .exp-card:hover {{ border-color: var(--accent-color); }}
        .exp-card-header {{ padding: 6px 10px; background: rgba(0,0,0,0.3); font-size: 0.75rem; font-weight: 700; color: var(--accent-purple); display: flex; justify-content: space-between; }}
        .exp-img-wrapper {{ width: 100%; aspect-ratio: 1/1; background: #000; overflow: hidden; }}
        .exp-img-wrapper img {{ width: 100%; height: 100%; object-fit: contain; }}

        /* Scores Banner */
        .score-pill {{ display: inline-flex; gap: 8px; background: rgba(74, 222, 128, 0.1); border: 1px solid rgba(74, 222, 128, 0.2); padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; color: var(--accent-green); }}

        /* Modal */
        .modal {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); backdrop-filter: blur(4px); z-index: 1000; align-items: center; justify-content: center; padding: 20px; }}
        .modal.active {{ display: flex; }}
        .modal-content {{ max-width: 90vw; max-height: 90vh; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; }}
        .modal-body {{ padding: 16px; background: #000; flex: 1; display: flex; align-items: center; justify-content: center; }}
        .modal-body img {{ max-width: 100%; max-height: 75vh; object-fit: contain; }}
        .modal-footer {{ padding: 12px 18px; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--card-border); }}
        .close-btn {{ background: var(--accent-color); color: #000; border: none; padding: 6px 16px; border-radius: 6px; font-weight: 700; cursor: pointer; }}
    </style>
</head>
<body>
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-header">
            <h1>🔬 Experiment Viewer</h1>
            <p>Unified Dataset & Generation Matrix</p>
        </div>
        <div class="section-label">Concepts (10개)</div>
        <ul class="nav-list" id="conceptNav"></ul>
    </div>

    <!-- Main Container -->
    <div class="main-container">
        <!-- Top Bar -->
        <div class="top-bar">
            <div class="concept-info">
                <span class="concept-title" id="conceptTitle">Loading...</span>
                <span class="concept-class" id="conceptClass"></span>
            </div>
            
            <div class="view-controls" id="expFilterGroup">
                <span style="font-size:0.78rem; color:var(--text-sub); margin-right:4px;">Filter Experiments:</span>
                <!-- Exp filter buttons dynamically inserted -->
            </div>
        </div>

        <!-- Scroll Content -->
        <div class="content-scroll">
            
            <!-- Reference Images (Dataset & Augmentation) -->
            <div class="box">
                <div class="box-header">
                    <div class="box-title">
                        <span>📸 원본(Dataset) 및 증강(Augmentation) 참조 이미지</span>
                    </div>
                </div>
                <div style="display:flex; flex-direction:column; gap:12px;">
                    <div>
                        <div style="font-size:0.8rem; font-weight:700; color:var(--text-sub); margin-bottom:6px;">• 원본 이미지 (dataset/) <span id="origCount" class="badge-count">0</span></div>
                        <div class="horizon-grid" id="origGrid"></div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem; font-weight:700; color:var(--text-sub); margin-bottom:6px;">• 증강 이미지 (augmentation/) <span id="augCount" class="badge-count">0</span></div>
                        <div class="horizon-grid" id="augGrid"></div>
                    </div>
                </div>
            </div>

            <!-- Prompt-wise Experiment Comparison Matrix -->
            <div class="box">
                <div class="box-header">
                    <div class="box-title">
                        <span>🎯 Iteration / Experiment별 Prompt 비교 Matrix</span>
                    </div>
                    <div id="scoresBanner" style="display:flex; gap:8px;"></div>
                </div>
                <div class="prompt-matrix" id="promptMatrix"></div>
            </div>

        </div>
    </div>

    <!-- Modal -->
    <div class="modal" id="modal" onclick="closeModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-body">
                <img id="modalImg" src="" alt="View">
            </div>
            <div class="modal-footer">
                <div>
                    <div id="modalTitle" style="font-weight:700; color:var(--text-main);"></div>
                    <div id="modalMeta" style="font-size:0.78rem; color:var(--text-sub);"></div>
                </div>
                <button class="close-btn" onclick="closeModal()">Close</button>
            </div>
        </div>
    </div>

    <script>
        const DATA = {json_data};
        let currentConcept = Object.keys(DATA.concepts)[0];
        let selectedExps = [...DATA.experiments]; // default all

        function init() {{
            renderSidebar();
            renderExpFilters();
            selectConcept(currentConcept);
        }}

        function renderSidebar() {{
            const nav = document.getElementById('conceptNav');
            nav.innerHTML = '';
            
            Object.keys(DATA.concepts).forEach(concept => {{
                const item = DATA.concepts[concept];
                const li = document.createElement('li');
                li.className = 'nav-item';
                
                const btn = document.createElement('button');
                btn.className = `nav-btn ${{concept === currentConcept ? 'active' : ''}}`;
                btn.onclick = () => selectConcept(concept);
                btn.innerHTML = `
                    <span>${{concept}}</span>
                    <span style="font-size:0.72rem; opacity:0.8;">${{item.orig_images.length}}장</span>
                `;
                li.appendChild(btn);
                nav.appendChild(li);
            }});
        }}

        function renderExpFilters() {{
            const group = document.getElementById('expFilterGroup');
            group.innerHTML = '<span style="font-size:0.78rem; color:var(--text-sub); margin-right:4px;">Iterations:</span>';
            
            // "ALL" Button
            const allBtn = document.createElement('button');
            allBtn.className = `filter-btn ${{selectedExps.length === DATA.experiments.length ? 'active' : ''}}`;
            allBtn.innerText = 'ALL';
            allBtn.onclick = () => {{
                selectedExps = [...DATA.experiments];
                renderExpFilters();
                renderMatrix();
            }};
            group.appendChild(allBtn);

            DATA.experiments.forEach(exp => {{
                const btn = document.createElement('button');
                const isSelected = selectedExps.includes(exp);
                btn.className = `filter-btn ${{isSelected && selectedExps.length !== DATA.experiments.length ? 'active' : ''}}`;
                btn.innerText = exp;
                btn.onclick = () => {{
                    if (selectedExps.includes(exp)) {{
                        if (selectedExps.length > 1) selectedExps = selectedExps.filter(e => e !== exp);
                    }} else {{
                        selectedExps.push(exp);
                    }}
                    renderExpFilters();
                    renderMatrix();
                }};
                group.appendChild(btn);
            }});
        }}

        function selectConcept(concept) {{
            currentConcept = concept;
            renderSidebar();
            
            const info = DATA.concepts[concept];
            document.getElementById('conceptTitle').innerText = concept;
            document.getElementById('conceptClass').innerText = `[class: "${{info.class_prompt}}"]`;
            
            renderReferenceImages(info);
            renderScores(concept);
            renderMatrix();
        }}

        function renderReferenceImages(info) {{
            // Original Grid
            const origGrid = document.getElementById('origGrid');
            document.getElementById('origCount').innerText = `${{info.orig_images.length}}장`;
            origGrid.innerHTML = info.orig_images.length === 0 ? '<span style="font-size:0.8rem; color:var(--text-sub);">없음</span>' : '';
            info.orig_images.forEach(img => {{
                const card = document.createElement('div');
                card.className = 'thumb-card';
                card.onclick = () => openModal(img.rel_path, img.filename, `dataset/${{currentConcept}} | ${{img.width}}x${{img.height}}`);
                card.innerHTML = `
                    <img class="thumb-img" src="${{img.rel_path}}" loading="lazy">
                    <div class="thumb-info" title="${{img.filename}}">${{img.filename}}</div>
                `;
                origGrid.appendChild(card);
            }});

            // Augmentation Grid
            const augGrid = document.getElementById('augGrid');
            document.getElementById('augCount').innerText = `${{info.aug_images.length}}장`;
            augGrid.innerHTML = info.aug_images.length === 0 ? '<span style="font-size:0.8rem; color:var(--text-sub);">증강 데이터 없음</span>' : '';
            info.aug_images.forEach(img => {{
                const card = document.createElement('div');
                card.className = 'thumb-card';
                card.onclick = () => openModal(img.rel_path, img.filename, `augmentation/${{currentConcept}} | ${{img.width}}x${{img.height}}`);
                card.innerHTML = `
                    <img class="thumb-img" src="${{img.rel_path}}" loading="lazy">
                    <div class="thumb-info" title="${{img.filename}}">${{img.filename}}</div>
                `;
                augGrid.appendChild(card);
            }});
        }}

        function renderScores(concept) {{
            const banner = document.getElementById('scoresBanner');
            banner.innerHTML = '';
            
            Object.keys(DATA.scores).forEach(exp => {{
                const scoreObj = DATA.scores[exp];
                if (scoreObj.per_concept_scores && scoreObj.per_concept_scores[concept]) {{
                    const s = scoreObj.per_concept_scores[concept];
                    const pill = document.createElement('div');
                    pill.className = 'score-pill';
                    pill.innerHTML = `<span><b>${{exp}}</b></span> | <span>CLIP-T: ${{s.t2i}}</span> | <span>CLIP-I: ${{s.i2i}}</span>`;
                    banner.appendChild(pill);
                }}
            }});
        }}

        function renderMatrix() {{
            const matrix = document.getElementById('promptMatrix');
            matrix.innerHTML = '';
            
            const info = DATA.concepts[currentConcept];
            const prompts = info.prompts;

            if (!prompts || prompts.length === 0) {{
                matrix.innerHTML = '<div style="color:var(--text-sub); padding:20px;">프롬프트 정보가 없습니다.</div>';
                return;
            }}

            prompts.forEach((pText, pIdx) => {{
                const row = document.createElement('div');
                row.className = 'prompt-row';
                
                // Prompt Header
                const rowHeader = document.createElement('div');
                rowHeader.className = 'prompt-row-header';
                rowHeader.innerHTML = `
                    <span class="prompt-idx-badge">Prompt #${{pIdx}}</span>
                    <span class="prompt-text">${{pText}}</span>
                `;
                row.appendChild(rowHeader);

                // Exp Columns
                const grid = document.createElement('div');
                grid.className = 'exp-comparison-grid';

                selectedExps.forEach(exp => {{
                    const expImgs = info.exp_generated[exp] || {{}};
                    const imgInfo = expImgs[pIdx];

                    const card = document.createElement('div');
                    card.className = 'exp-card';

                    if (imgInfo) {{
                        card.onclick = () => openModal(imgInfo.rel_path, `${{exp}} - Prompt #${{pIdx}}`, pText);
                        card.innerHTML = `
                            <div class="exp-card-header">
                                <span>${{exp}}</span>
                                <span style="color:var(--accent-color);">#${{pIdx}}</span>
                            </div>
                            <div class="exp-img-wrapper">
                                <img src="${{imgInfo.rel_path}}" loading="lazy">
                            </div>
                        `;
                    }} else {{
                        card.innerHTML = `
                            <div class="exp-card-header">
                                <span>${{exp}}</span>
                                <span>N/A</span>
                            </div>
                            <div class="exp-img-wrapper" style="display:flex; align-items:center; justify-content:center; color:var(--text-sub); font-size:0.75rem;">
                                생성 이미지 없음
                            </div>
                        `;
                    }}
                    grid.appendChild(card);
                }});

                row.appendChild(grid);
                matrix.appendChild(row);
            }});
        }}

        function openModal(src, title, meta) {{
            document.getElementById('modalImg').src = src;
            document.getElementById('modalTitle').innerText = title;
            document.getElementById('modalMeta').innerText = meta;
            document.getElementById('modal').classList.add('active');
        }}

        function closeModal(e) {{
            document.getElementById('modal').classList.remove('active');
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeModal();
        }});

        init();
    </script>
</body>
</html>
"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✓ Experiment Viewer HTML이 성공적으로 생성되었습니다: {output_html_path}")


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
    else:
        print(f"💡 브라우저에서 파일 직접 열기: file:///{out_path.replace('\\', '/')}")


if __name__ == "__main__":
    main()
