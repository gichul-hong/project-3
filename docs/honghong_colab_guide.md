# honghong Personalization — Colab 실행 가이드

## 🎯 목표
이번 과제 파이프라인(Exp-13 Controlled ODE + Spherical Blend + MMR 선별기)을 **실제 어린이 사진(honghong)**에 적용하여 새로운 씬을 생성합니다.

---

## 📋 전제 조건

| 항목 | 내용 |
|:--|:--|
| **GPU** | A100 / L4 / T4 중 1개 (Colab Pro 권장, T4도 동작하나 15~30분 소요) |
| **스토리지** | Google Drive 마운트 (체크포인트 약 2GB) |
| **파이프라인** | project-3 레포 (이미 있음) |

---

## 🚀 Step 1 — Colab 환경 세팅

```python
# 셀 1: GPU 확인
!nvidia-smi
import torch
print("CUDA:", torch.cuda.is_available(), "| Device:", torch.cuda.get_device_name(0))
```

```python
# 셀 2: Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')
```

```python
# 셀 3: project-3 위치 확인 (Drive에서 복사하거나 git clone)
import os

# 옵션 A: Drive에 프로젝트가 있는 경우
PROJECT_DIR = "/content/drive/MyDrive/project-3"   # ← 본인 경로로 수정

# 옵션 B: Drive에서 /content/로 복사 (속도 향상)
# !cp -r "/content/drive/MyDrive/project-3" /content/
# PROJECT_DIR = "/content/project-3"

print("프로젝트 경로:", PROJECT_DIR)
os.listdir(PROJECT_DIR)[:10]
```

```python
# 셀 4: 의존성 설치
!pip install -q diffusers transformers accelerate peft tqdm pillow
```

---

## 🖼️ Step 2 — dataset/honghong 사진 확인

```python
# 셀 5: 사진 목록 및 미리보기
import os
from PIL import Image
import matplotlib.pyplot as plt

honghong_dir = os.path.join(PROJECT_DIR, "dataset/honghong")
img_files = sorted([f for f in os.listdir(honghong_dir)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))])
print(f"총 {len(img_files)}장 발견:", img_files)

# 처음 6장 미리보기
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, fname in zip(axes.flat, img_files[:6]):
    img = Image.open(os.path.join(honghong_dir, fname)).convert("RGB")
    ax.imshow(img); ax.set_title(fname[:20], fontsize=9); ax.axis('off')
plt.tight_layout(); plt.show()
```

> [!WARNING]
> **HEIC 파일 주의**: `IMG_6445.HEIC` 파일은 PIL이 직접 읽지 못합니다. 아래 셀로 미리 변환하세요.

```python
# 셀 5-b: HEIC → JPG 변환 (HEIC 파일이 있는 경우만 실행)
!pip install -q pillow-heif

import os
from pathlib import Path
from pillow_heif import register_heif_opener
from PIL import Image
register_heif_opener()

honghong_dir = os.path.join(PROJECT_DIR, "dataset/honghong")
heic_files = [f for f in os.listdir(honghong_dir) if f.lower().endswith('.heic')]

for fname in heic_files:
    src = os.path.join(honghong_dir, fname)
    dst = os.path.join(honghong_dir, Path(fname).stem + ".jpg")
    img = Image.open(src).convert("RGB")
    img.save(dst, "JPEG", quality=95)
    os.remove(src)   # 원본 HEIC 삭제
    print(f"변환 완료: {fname} → {Path(fname).stem}.jpg")

print("HEIC 변환 완료!")
```


---

## ⚡ Step 3 — 파이프라인 실행

### 옵션 A: LoRA 없이 (base SD3.5만으로 빠르게 확인)

> **소요 시간**: T4 기준 약 8~12분

```python
# 셀 6-A: base 모델로 실행 (LoRA 없이)
!python {PROJECT_DIR}/run_honghong_exp13.py \
    --root {PROJECT_DIR} \
    --output_dir {PROJECT_DIR}/experiments/honghong_exp13 \
    --tau 0.58 \
    --eta 0.70 \
    --candidates 4 \
    --steps 28
```

---

### 옵션 B: LoRA 먼저 학습 후 실행 (더 좋은 결과)

> **학습 소요**: A100 기준 약 15~20분 / T4 기준 50~70분

```python
# 셀 6-B-1: honghong LoRA 학습
LORA_OUT = f"{PROJECT_DIR}/checkpoints/exp05_lora_hq/lora_honghong"

!python {PROJECT_DIR}/train_lora_sd3.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-3.5-medium" \
    --instance_data_dir {PROJECT_DIR}/dataset/honghong \
    --instance_prompt "photo of a sks child" \
    --output_dir {LORA_OUT} \
    --rank 64 \
    --learning_rate 5e-5 \
    --max_train_steps 800 \
    --train_text_encoder \
    --resolution 512 \
    --seed 42
```

```python
# 셀 6-B-2: LoRA 적용 후 생성
!python {PROJECT_DIR}/run_honghong_exp13.py \
    --root {PROJECT_DIR} \
    --checkpoints_dir {PROJECT_DIR}/checkpoints/exp05_lora_hq \
    --output_dir {PROJECT_DIR}/experiments/honghong_exp13 \
    --tau 0.58 \
    --eta 0.70 \
    --candidates 4 \
    --steps 28
```

---

### 옵션 C: DreamBooth 학습 후 실행 (최고 품질)

> **학습 소요**: A100 기준 약 25분

```python
# 셀 6-C-1: DreamBooth-LoRA 학습
DB_OUT = f"{PROJECT_DIR}/checkpoints/exp08_dreambooth_lora/lora_honghong"

!python {PROJECT_DIR}/train_dreambooth_sd3.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-3.5-medium" \
    --instance_data_dir {PROJECT_DIR}/dataset/honghong \
    --instance_prompt "a photo of sks child" \
    --class_prompt "a photo of a child" \
    --output_dir {DB_OUT} \
    --rank 64 \
    --learning_rate 5e-5 \
    --max_train_steps 800 \
    --with_prior_preservation \
    --prior_loss_weight 0.3 \
    --num_class_images 200 \
    --resolution 512 \
    --seed 42
```

```python
# 셀 6-C-2: DreamBooth LoRA 적용 후 생성
!python {PROJECT_DIR}/run_honghong_exp13.py \
    --root {PROJECT_DIR} \
    --checkpoints_dir {PROJECT_DIR}/checkpoints/exp08_dreambooth_lora \
    --output_dir {PROJECT_DIR}/experiments/honghong_exp13 \
    --tau 0.58 \
    --eta 0.70 \
    --candidates 4 \
    --steps 28
```

---

## 📊 Step 4 — 결과 확인 및 시각화

```python
# 셀 7: 생성 이미지 전체 시각화
import json, os
from PIL import Image
import matplotlib.pyplot as plt

out_dir = f"{PROJECT_DIR}/experiments/honghong_exp13/honghong"
prompt_file = f"{PROJECT_DIR}/prompt/honghong.txt"

with open(prompt_file) as f:
    prompts = [l.strip().replace("{}", "child") for l in f if l.strip()]

imgs = [Image.open(os.path.join(out_dir, f"{i}.png")) for i in range(10)]

fig, axes = plt.subplots(2, 5, figsize=(20, 9))
for ax, img, prompt in zip(axes.flat, imgs, prompts):
    ax.imshow(img)
    ax.set_title(prompt[:45], fontsize=8, wrap=True)
    ax.axis('off')
plt.suptitle("🍀 honghong — Exp-13 Personalization Results", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/experiments/honghong_exp13/honghong_gallery.png",
            bbox_inches='tight', dpi=150)
plt.show()
print("갤러리 저장 완료!")
```

```python
# 셀 8: 점수 확인
result_path = f"{PROJECT_DIR}/experiments/honghong_exp13/honghong_result.json"
with open(result_path) as f:
    result = json.load(f)

avg = result['average']
print(f"{'='*50}")
print(f"🍀 honghong Personalization 결과")
print(f"{'='*50}")
print(f"  CLIP-T (텍스트 충실도): {avg['clip_t']:.4f}")
print(f"  CLIP-I (정체성 보존):   {avg['clip_i']:.4f}")
print(f"  Total Score:          {avg['total']:.4f}")
print(f"  사용된 LoRA:           {result['lora_used']}")
print(f"  τ={result['tau']}, η={result['eta']}")
print(f"{'='*50}")

# 프롬프트별 점수
print("\n프롬프트별 상세 점수:")
for r in result['per_prompt']:
    print(f"  [{r['prompt_idx']}] T={r['clip_t']:.3f}  I={r['clip_i']:.3f}  "
          f"Tot={r['clip_t']+r['clip_i']:.3f}  |  {r['prompt'][:50]}")
```

```python
# 셀 9: 원본 vs 생성 비교 (Best 3장)
import matplotlib.pyplot as plt
from PIL import Image
import os

# 원본 레퍼런스 (첫 번째 사진)
honghong_dir = f"{PROJECT_DIR}/dataset/honghong"
ref_files = sorted([f for f in os.listdir(honghong_dir)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))])[:3]
gen_dir = f"{PROJECT_DIR}/experiments/honghong_exp13/honghong"

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 상단: 원본 레퍼런스
for ax, fname in zip(axes[0], ref_files):
    img = Image.open(os.path.join(honghong_dir, fname)).convert("RGB")
    ax.imshow(img); ax.set_title(f"📷 원본: {fname[:18]}", fontsize=9); ax.axis('off')

# 하단: 생성 결과 (0, 3, 8번 프롬프트)
for ax, idx in zip(axes[1], [0, 3, 8]):
    img = Image.open(os.path.join(gen_dir, f"{idx}.png"))
    title = prompts[idx][:40] + "..."
    ax.imshow(img); ax.set_title(f"🎨 생성 [{idx}]: {title}", fontsize=8); ax.axis('off')

plt.suptitle("🍀 honghong: 원본 사진 vs AI 생성 이미지", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/experiments/honghong_exp13/honghong_comparison.png",
            bbox_inches='tight', dpi=150)
plt.show()
```

---

## 💡 파라미터 조절 팁

| 파라미터 | 기본값 | 높이면 | 낮추면 |
|:--|:--:|:--|:--|
| `--tau` | 0.58 | 원본 얼굴 더 강하게 유지 | 배경 변환 더 자유롭게 |
| `--eta` | 0.70 | 정체성 임베딩 강화 | 프롬프트 반영도 ↑ |
| `--candidates` | 4 | 품질 ↑ (시간 ↑) | 빠른 실험 |
| `--guidance` | 7.0 | 프롬프트 충실도 ↑ | 이미지 자연스러움 ↑ |

### 🔧 어린이 얼굴 최적화 권장 설정

```bash
# 얼굴 정체성 중시 (CLIP-I 극대화)
--tau 0.65 --eta 0.78

# 배경 변환 중시 (CLIP-T 극대화)  
--tau 0.52 --eta 0.62

# 균형 (기본)
--tau 0.58 --eta 0.70
```

---

## 📁 생성되는 파일 구조

```
experiments/honghong_exp13/
├── honghong/
│   ├── 0.png  ~ 9.png          # 최종 선별 이미지 10장
│   └── (프롬프트 순서와 1:1 대응)
├── candidates/
│   └── honghong/
│       ├── p0_c0.png ~ p9_c3.png  # 후보 40장 (10×4)
├── honghong_result.json            # CLIP-T/I 점수 및 선별 기록
├── honghong_gallery.png            # 10장 갤러리 이미지
└── honghong_comparison.png         # 원본 vs 생성 비교
```

---

## ✅ 완료 체크리스트

- [ ] GPU 환경 확인 (A100 / L4 / T4)
- [ ] `dataset/honghong/` 경로에 사진 20장 존재 확인
- [ ] `prompt/honghong.txt` 파일 확인 (10개 프롬프트)
- [ ] 옵션 선택: A (빠른 확인) / B (LoRA) / C (DreamBooth)
- [ ] 생성 완료 후 갤러리 이미지 확인
- [ ] `honghong_result.json`에서 CLIP 점수 확인
- [ ] 발표 자료에 원본 vs 생성 비교 슬라이드 추가
