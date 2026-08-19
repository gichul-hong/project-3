"""
Dataset & Augmentation Web Viewer Generator
-------------------------------------------
이 스크립트는 dataset/, augmentation/, prompt/ 폴더의 데이터를 자동으로 스캔하여
웹 브라우저에서 편리하게 데이터셋 및 증강 이미지를 조망할 수 있는
인터랙티브 웹 대시보드(dataset_viewer.html)를 생성합니다.

사용법:
    python generate_dataset_viewer.py [--server] [--port 8000]
"""

import argparse
import glob
import json
import os
import http.server
import socketserver
import webbrowser
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


def get_image_info(img_path):
    """이미지 해상도, 용량, 상대 경로 추출"""
    rel_path = os.path.relpath(img_path, r"C:\hong\project-3").replace("\\", "/")
    file_size_kb = round(os.path.getsize(img_path) / 1024, 1)
    filename = os.path.basename(img_path)
    
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            format_name = img.format
    except Exception:
        width, height = 0, 0
        format_name = "UNKNOWN"
        
    return {
        "filename": filename,
        "rel_path": rel_path,
        "width": width,
        "height": height,
        "size_kb": file_size_kb,
        "format": format_name
    }


def scan_dataset(root_dir):
    data = {}
    
    for concept, class_prompt in CLASS_PROMPT.items():
        # 원본 데이터 스캔
        orig_dir = os.path.join(root_dir, "dataset", concept)
        orig_imgs = []
        if os.path.exists(orig_dir):
            paths = sorted(
                glob.glob(os.path.join(orig_dir, "*.png")) +
                glob.glob(os.path.join(orig_dir, "*.jpg")) +
                glob.glob(os.path.join(orig_dir, "*.jpeg"))
            )
            orig_imgs = [get_image_info(p) for p in paths]

        # 증강 데이터 스캔
        aug_dir = os.path.join(root_dir, "augmentation", concept)
        aug_imgs = []
        if os.path.exists(aug_dir):
            paths = sorted(
                glob.glob(os.path.join(aug_dir, "*.png")) +
                glob.glob(os.path.join(aug_dir, "*.jpg")) +
                glob.glob(os.path.join(aug_dir, "*.jpeg"))
            )
            aug_imgs = [get_image_info(p) for p in paths]

        # 프롬프트 스캔
        prompt_file = os.path.join(root_dir, "prompt", f"{concept}.txt")
        prompts = []
        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompts = [
                    l.strip().replace("{}", class_prompt)
                    for l in f.readlines() if l.strip()
                ]

        data[concept] = {
            "class_prompt": class_prompt,
            "orig_count": len(orig_imgs),
            "aug_count": len(aug_imgs),
            "orig_images": orig_imgs,
            "aug_images": aug_imgs,
            "prompts": prompts
        }
        
    return data


def generate_html(data, output_html_path):
    json_data = json.dumps(data, ensure_ascii=False)
    
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VERILUX Dataset & Augmentation Viewer</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #38bdf8;
            --accent-hover: #0284c7;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --border-color: #334155;
            --badge-bg: #334155;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }}
        body {{ background-color: var(--bg-color); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }}

        /* Sidebar Navigation */
        .sidebar {{ width: 280px; background-color: #090d16; border-right: 1px solid var(--border-color); display: flex; flex-direction: column; }}
        .sidebar-header {{ padding: 20px; border-bottom: 1px solid var(--border-color); }}
        .sidebar-header h1 {{ font-size: 1.25rem; font-weight: 700; color: var(--accent-color); display: flex; align-items: center; gap: 8px; }}
        .sidebar-header p {{ font-size: 0.8rem; color: var(--text-sub); margin-top: 4px; }}
        .nav-list {{ list-style: none; overflow-y: auto; flex: 1; padding: 12px 8px; }}
        .nav-item {{ margin-bottom: 4px; }}
        .nav-btn {{ width: 100%; text-align: left; padding: 12px 14px; background: transparent; border: none; color: var(--text-sub); border-radius: 8px; cursor: pointer; font-size: 0.9rem; font-weight: 600; display: flex; justify-content: space-between; align-items: center; transition: all 0.2s; }}
        .nav-btn:hover {{ background-color: rgba(56, 189, 248, 0.1); color: var(--text-main); }}
        .nav-btn.active {{ background-color: var(--accent-color); color: #000; font-weight: 700; }}
        .badge {{ font-size: 0.75rem; padding: 2px 8px; border-radius: 12px; background-color: rgba(255, 255, 255, 0.15); color: inherit; }}

        /* Main Content Area */
        .main-container {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
        .top-bar {{ padding: 16px 24px; background-color: var(--card-bg); border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; }}
        .concept-title {{ font-size: 1.4rem; font-weight: 700; color: var(--text-main); }}
        .concept-class {{ font-size: 0.9rem; color: var(--accent-color); font-weight: 500; margin-left: 8px; }}
        
        .tab-group {{ display: flex; background-color: var(--bg-color); padding: 4px; border-radius: 8px; border: 1px solid var(--border-color); }}
        .tab-btn {{ padding: 8px 16px; border: none; background: transparent; color: var(--text-sub); border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: all 0.2s; }}
        .tab-btn.active {{ background-color: var(--accent-color); color: #000; }}

        .content-scroll {{ flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 24px; }}
        
        /* Prompts Section */
        .section-box {{ background-color: var(--card-bg); border-radius: 12px; border: 1px solid var(--border-color); padding: 18px 20px; }}
        .section-title {{ font-size: 1rem; font-weight: 700; color: var(--accent-color); margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; }}
        .prompt-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 8px; }}
        .prompt-item {{ background-color: rgba(15, 23, 42, 0.6); padding: 8px 12px; border-radius: 6px; font-size: 0.82rem; color: var(--text-sub); border: 1px solid rgba(255, 255, 255, 0.05); font-family: monospace; display: flex; gap: 8px; }}
        .prompt-idx {{ color: var(--accent-color); font-weight: 700; min-width: 20px; }}

        /* Image Grid */
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }}
        .img-card {{ background-color: var(--card-bg); border: 1px solid var(--border-color); border-radius: 10px; overflow: hidden; transition: transform 0.2s, border-color 0.2s; cursor: pointer; position: relative; }}
        .img-card:hover {{ transform: translateY(-4px); border-color: var(--accent-color); box-shadow: 0 10px 20px rgba(0,0,0,0.3); }}
        .img-wrapper {{ width: 100%; aspect-ratio: 1/1; background-color: #000; overflow: hidden; position: relative; display: flex; align-items: center; justify-content: center; }}
        .img-wrapper img {{ width: 100%; height: 100%; object-fit: contain; }}
        .img-meta {{ padding: 10px 12px; font-size: 0.78rem; }}
        .img-name {{ font-weight: 700; color: var(--text-main); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 4px; }}
        .img-details {{ display: flex; justify-content: space-between; color: var(--text-sub); font-size: 0.72rem; }}

        /* Lightbox Modal */
        .modal {{ display: none; position: fixed; inset: 0; background-color: rgba(0,0,0,0.85); backdrop-filter: blur(4px); z-index: 1000; align-items: center; justify-content: center; padding: 24px; }}
        .modal.active {{ display: flex; }}
        .modal-content {{ max-width: 90vw; max-height: 90vh; background-color: var(--card-bg); border-radius: 12px; border: 1px solid var(--border-color); overflow: hidden; display: flex; flex-direction: column; }}
        .modal-img-container {{ flex: 1; overflow: hidden; display: flex; align-items: center; justify-content: center; background-color: #000; padding: 12px; }}
        .modal-img-container img {{ max-width: 100%; max-height: 75vh; object-fit: contain; }}
        .modal-footer {{ padding: 16px 20px; border-top: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; }}
        .close-btn {{ background-color: var(--accent-color); color: #000; border: none; padding: 8px 16px; border-radius: 6px; font-weight: 700; cursor: pointer; }}
    </style>
</head>
<body>
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-header">
            <h1>🔍 VERILUX Viewer</h1>
            <p>Subject Dataset & Augmentation Inspection</p>
        </div>
        <ul class="nav-list" id="navList"></ul>
    </div>

    <!-- Main Content -->
    <div class="main-container">
        <div class="top-bar">
            <div>
                <span class="concept-title" id="conceptTitle">Loading...</span>
                <span class="concept-class" id="conceptClass"></span>
            </div>
            <div class="tab-group">
                <button class="tab-btn active" id="tabOrig" onclick="switchDataset('orig')">Original Dataset (dataset/)</button>
                <button class="tab-btn" id="tabAug" onclick="switchDataset('aug')">Augmented (augmentation/)</button>
            </div>
        </div>

        <div class="content-scroll">
            <!-- Prompts Box -->
            <div class="section-box">
                <div class="section-title">
                    <span>📝 Test Prompts (10개)</span>
                    <span style="font-size: 0.8rem; font-weight: 400; color: var(--text-sub);">Evaluation target prompts</span>
                </div>
                <div class="prompt-grid" id="promptGrid"></div>
            </div>

            <!-- Image Grid Section -->
            <div class="section-box">
                <div class="section-title">
                    <span id="imageSectionTitle">🖼️ Images</span>
                    <span style="font-size: 0.8rem; color: var(--text-sub);" id="imageCountBadge">0 images</span>
                </div>
                <div class="image-grid" id="imageGrid"></div>
            </div>
        </div>
    </div>

    <!-- Modal Lightbox -->
    <div class="modal" id="lightboxModal" onclick="closeModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-img-container">
                <img id="modalImg" src="" alt="Full view">
            </div>
            <div class="modal-footer">
                <div>
                    <div style="font-weight: 700; color: var(--text-main);" id="modalName"></div>
                    <div style="font-size: 0.8rem; color: var(--text-sub);" id="modalMeta"></div>
                </div>
                <button class="close-btn" onclick="closeModal()">Close</button>
            </div>
        </div>
    </div>

    <script>
        const DATA = {json_data};
        let currentConcept = Object.keys(DATA)[0];
        let currentView = 'orig'; // 'orig' or 'aug'

        function init() {{
            renderSidebar();
            selectConcept(currentConcept);
        }}

        function renderSidebar() {{
            const navList = document.getElementById('navList');
            navList.innerHTML = '';
            
            Object.keys(DATA).forEach(concept => {{
                const item = DATA[concept];
                const li = document.createElement('li');
                li.className = 'nav-item';
                
                const btn = document.createElement('button');
                btn.className = `nav-btn ${{concept === currentConcept ? 'active' : ''}}`;
                btn.onclick = () => selectConcept(concept);
                btn.innerHTML = `
                    <span>${{concept}}</span>
                    <span class="badge">${{item.orig_count}} / ${{item.aug_count}}장</span>
                `;
                li.appendChild(btn);
                navList.appendChild(li);
            }});
        }}

        function selectConcept(concept) {{
            currentConcept = concept;
            renderSidebar();
            
            const info = DATA[concept];
            document.getElementById('conceptTitle').innerText = concept;
            document.getElementById('conceptClass').innerText = `[class prompt: "${{info.class_prompt}}"]`;
            
            renderPrompts(info.prompts);
            renderImages();
        }}

        function switchDataset(type) {{
            currentView = type;
            document.getElementById('tabOrig').className = `tab-btn ${{type === 'orig' ? 'active' : ''}}`;
            document.getElementById('tabAug').className = `tab-btn ${{type === 'aug' ? 'active' : ''}}`;
            renderImages();
        }}

        function renderPrompts(prompts) {{
            const grid = document.getElementById('promptGrid');
            grid.innerHTML = '';
            prompts.forEach((p, idx) => {{
                const div = document.createElement('div');
                div.className = 'prompt-item';
                div.innerHTML = `<span class="prompt-idx">${{idx}}</span> <span>${{p}}</span>`;
                grid.appendChild(div);
            }});
        }}

        function renderImages() {{
            const info = DATA[currentConcept];
            const images = currentView === 'orig' ? info.orig_images : info.aug_images;
            const grid = document.getElementById('imageGrid');
            grid.innerHTML = '';

            document.getElementById('imageSectionTitle').innerText = currentView === 'orig' 
                ? '🖼️ Original Dataset (dataset/)' 
                : '✨ Augmented Dataset (augmentation/)';
            document.getElementById('imageCountBadge').innerText = `총 ${{images.length}}장`;

            if (images.length === 0) {{
                grid.innerHTML = '<div style="grid-column: 1/-1; color: var(--text-sub); text-align: center; padding: 40px;">이미지가 없습니다.</div>';
                return;
            }}

            images.forEach(img => {{
                const card = document.createElement('div');
                card.className = 'img-card';
                card.onclick = () => openModal(img);
                card.innerHTML = `
                    <div class="img-wrapper">
                        <img src="${{img.rel_path}}" alt="${{img.filename}}" loading="lazy">
                    </div>
                    <div class="img-meta">
                        <div class="img-name" title="${{img.filename}}">${{img.filename}}</div>
                        <div class="img-details">
                            <span>${{img.width}}x${{img.height}}</span>
                            <span>${{img.size_kb}} KB</span>
                        </div>
                    </div>
                `;
                grid.appendChild(card);
            }});
        }}

        function openModal(img) {{
            document.getElementById('modalImg').src = img.rel_path;
            document.getElementById('modalName').innerText = img.filename;
            document.getElementById('modalMeta').innerText = `경로: ${{img.rel_path}} | 해상도: ${{img.width}}x${{img.height}} | 용량: ${{img.size_kb}} KB | 포맷: ${{img.format}}`;
            document.getElementById('lightboxModal').classList.add('active');
        }}

        function closeModal(e) {{
            document.getElementById('lightboxModal').classList.remove('active');
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
        
    print(f"✓ Dataset Viewer HTML이 성공적으로 생성되었습니다: {output_html_path}")


def start_server(html_path, host="0.0.0.0", port=8000):
    root_dir = os.path.dirname(os.path.abspath(html_path))
    os.chdir(root_dir)
    
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer((host, port), handler)
    
    file_name = os.path.basename(html_path)
    ext_url = f"http://147.47.201.63:{port}/{file_name}"
    local_url = f"http://localhost:{port}/{file_name}"
    
    print(f"\n🚀 웹 서버 구동 중 (외부 접속 가능):")
    print(f"  • 외부 IP 접속: {ext_url}")
    print(f"  • 로컬 접속:    {local_url}")
    print("종료하려면 Ctrl+C를 누르세요.\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n서버가 종료되었습니다.")
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Dataset Viewer Generator")
    parser.add_argument("--root", type=str, default=r"C:\hong\project-3", help="프로젝트 루트 폴더")
    parser.add_argument("--out", type=str, default="dataset_viewer.html", help="생성할 HTML 파일명")
    parser.add_argument("--server", action="store_true", help="생성 후 HTTP 서버 자동 구동")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="바인딩 호스트 주소 (기본 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="웹 서버 포트 번호")

    args = parser.parse_args()
    
    print("데이터셋 스캔 중...")
    data = scan_dataset(args.root)
    out_path = os.path.join(args.root, args.out)
    generate_html(data, out_path)
    
    if args.server:
        start_server(out_path, host=args.host, port=args.port)
    else:
        print(f"💡 브라우저에서 direct로 바로 열기: file:///{out_path.replace('\\', '/')}")


if __name__ == "__main__":
    main()
