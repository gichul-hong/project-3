# Subject-driven Customization 데이터 증강 계획서 (AUGMENTATION_PLAN.md)

> **프로젝트**: VERILUX (SD3.5-medium 기반 Subject-driven Customization)  
> **관련 스크립트**: [`augment_dataset.py`](file:///content/project-3/augment_dataset.py)  
> **출력 디렉터리**: [`augmentation/`](file:///content/project-3/augmentation)  
> **최종 현행화 일자**: 2026-08-19 (v2 - rembg GPU 가속 + 캡션 생성 + 선택적 좌우반전 반영)  

---

## 1. 개요 및 목적

### 1.1 데이터셋 한계 및 과제
- **데이터 부족**: CustomConcept101 10개 서브젝트의 레퍼런스는 서브젝트당 4~15장에 불과하여 학습 시 과적합(Overfitting) 발생 가능.
- **해상도 및 종횡비 불균형**: 700px ~ 9000px 초고해상도 및 직사각형 비율 혼재로 인해 왜곡 없는 전처리 필수.
- **배경 간섭(Background Entanglement/Bleeding)**: 레퍼런스 이미지의 고유 배경 정보가 서브젝트 정체성과 함께 학습되어 새로운 프롬프트(`"in times square"`, `"on the beach"`) 생성 시 배경 조화 및 다양성을 저해함.

### 1.2 증강(Augmentation)의 3대 목표
1. **정체성 유지 (Identity Preservation - CLIP-I 극대화)**: 레퍼런스 객체의 고유 디테일, 텍스처, 색상 정체성을 손상시키지 않고 보존.
2. **프롬프트 준수 (Prompt Fidelity - CLIP-T 향상)**: 배경 제거(`rembg`)를 통한 서브젝트-배경 분리로 신규 프롬프트 배경 렌더링 자유도 확보.
3. **학습 데이터 다양성 및 서브젝트별 맞춤 증강**: 비대칭 사물(기타, 가방 등)의 왜곡을 방지하는 선택적 반전(Selective Flip) 및 조명/대비 미세 변형 적용.

---

## 2. 증강 전략 및 세부 기법 (v2 Pipeline)

### 2.1 적용 기법 세부 사양

| 기법 | 생성 파일 접미사 | 적용 원리 및 파라미터 | 도입 목적 및 효과 |
|---|---|---|---|
| **Center Fit & Padding** | `_std.png` | 종횡비 유지 축소 후 512x512 흰색 캔버스 중앙 배치 (Lanczos) | 객체 찌그러짐 방지, SD3.5 표준 입력 규격화 |
| **Selective Horizontal Flip** | `_flip.png` | 좌우 대칭 반전 (단, 비대칭/방향성 서브젝트 제외) | 좌/우 한쪽 앵글 편향 극복 및 포즈 유연성 강화 |
| **Light & Contrast Adj.** | `_light.png` | 밝기(Brightness) 1.05x, 대비(Contrast) 1.10x 미세 조정 | 다양한 조명 환경에 대한 강건성(Robustness) 확보 |
| **AI Background Removal** | `_nobg.png` | `bria-rmbg-2.0.onnx` 기반 객체 누끼 분리 후 흰색 패딩 | 배경 정보 얽힘 차단, CLIP-T 및 정성 다양성 급상승 |
| **Background Removed Flip** | `_nobg_flip.png` | 배경 제거 객체의 좌우 반전 (선택적 적용) | 무배경 객체의 다양한 앵글 학습 데이터 추가 확보 |

### 2.2 특수 규칙 및 예외 처리

1. **Selective Flip (좌우 반전 제외 대상 - `SKIP_FLIP_CONCEPTS`)**:
   - 대상: `instrument_music2` (기타 현/헤드 방향), `luggage_backpack1` (비대칭 스트랩/로고), `transport_tank` (포탑/무한궤도 비대칭), `wearable_jacket1` (지퍼/포켓 비대칭)
   - 사유: 비대칭성이 뚜렷한 사물에 좌우 반전을 적용할 경우 비현실적인 가짜 정체성(Hallucinated Symmetry)이 학습되는 문제를 원천 차단.

2. **Selective rembg (배경 제거 제외 대상 - `SKIP_REMBG_CONCEPTS`)**:
   - 대상: `scene_waterfall` (폭포 풍경)
   - 사유: 자연 풍경 서브젝트는 배경과 주 객체의 경계가 없으며 전체 프레임이 서브젝트이므로 배경 제거 시 이미지 손실 발생.

3. **자동 캡션 및 메타데이터 페어링**:
   - 모든 증강 이미지마다 동일한 이름의 `.txt` 캡션 파일 생성 (`a photo of sks <class_prompt>`).
   - 서브젝트 폴더별로 Hugging Face / Diffusers 호환 `metadata.jsonl` 파일 자동 생성.

---

## 3. 서브젝트별 증강 수량 현황 (총 286장)

> `--use_rembg` 옵션 적용 시 기준 (기본 모드 실행 시 총 187장)

| 번호 | 서브젝트 폴더명 | Class Prompt | 원본 수 | 증강 후 수량 | 적용 기법 규칙 |
|:---:|---|---|:---:|:---:|---|
| 1 | `actionfigure_2` | action figure | 6장 | **30장** (5배) | CenterFit, Flip, Light, rembg, rembg_flip |
| 2 | `decoritems_woodenpot` | wooden pot | 4장 | **20장** (5배) | CenterFit, Flip, Light, rembg, rembg_flip |
| 3 | `furniture_sofa2` | sofa | 4장 | **20장** (5배) | CenterFit, Flip, Light, rembg, rembg_flip |
| 4 | `instrument_music2` | guitar | 6장 | **18장** (3배) | CenterFit, Light, rembg `[Flip 제외]` |
| 5 | `luggage_backpack1` | backpack | 5장 | **15장** (3배) | CenterFit, Light, rembg `[Flip 제외]` |
| 6 | `person_3` | person | 15장 | **75장** (5배) | CenterFit, Flip, Light, rembg, rembg_flip |
| 7 | `pet_cat5` | cat | 9장 | **45장** (5배) | CenterFit, Flip, Light, rembg, rembg_flip |
| 8 | `scene_waterfall` | waterfall | 9장 | **27장** (3배) | CenterFit, Flip, Light `[rembg 제외]` |
| 9 | `transport_tank` | tank | 7장 | **21장** (3배) | CenterFit, Light, rembg `[Flip 제외]` |
| 10 | `wearable_jacket1` | jacket | 5장 | **15장** (3배) | CenterFit, Light, rembg `[Flip 제외]` |
| **합계** | **10개 서브젝트** | - | **70장** | **총 286장** | **-** |

---

## 4. 환경 설정 및 실행 방법

### 4.1 실행 환경 및 가속 설정 (CUDA Runtime)
- **GPU 가속**: ONNX Runtime GPU (`CUDAExecutionProvider`) 연동
- **환경 구성**: PyTorch 2.11.0 + CUDA 12.8 호환 `onnxruntime-gpu==1.20.0` 패키지 구성으로 고속 배경 분리 추론 수행 (장당 1초 이내)

### 4.2 실행 명령어

```bash
# 1) 전체 10개 서브젝트 일괄 증강 (rembg 포함, 권장)
python augment_dataset.py --dataset ./dataset --output ./augmentation --concept all --use_rembg

# 2) 특정 서브젝트만 선택 증강
python augment_dataset.py --dataset ./dataset --output ./augmentation --concept actionfigure_2 --use_rembg

# 3) 배경 제거 없이 기본 기하학/대비 증강만 수행 (총 187장)
python augment_dataset.py --dataset ./dataset --output ./augmentation --concept all
```

### 4.3 출력 디렉터리 구조

```text
augmentation/
├── actionfigure_2/
│   ├── 00_0_std.png          # 512x512 중앙 패딩 표준
│   ├── 00_0_std.txt          # "a photo of sks action figure"
│   ├── 00_0_flip.png         # 좌우 대칭 반전
│   ├── 00_0_flip.txt
│   ├── 00_0_light.png        # 1.05x 밝기, 1.10x 대비 조정
│   ├── 00_0_light.txt
│   ├── 00_0_nobg.png         # rembg 배경 제거
│   ├── 00_0_nobg.txt
│   ├── 00_0_nobg_flip.png    # rembg 배경 제거 + 좌우 반전
│   ├── 00_0_nobg_flip.txt
│   ├── metadata.jsonl        # Diffusers 학습용 jsonl 메타데이터
│   └── ... (총 30장 이미지 + 30개 txt)
├── decoritems_woodenpot/     (총 20장)
├── furniture_sofa2/          (총 20장)
├── instrument_music2/        (총 18장)
├── luggage_backpack1/        (총 15장)
├── person_3/                 (총 75장)
├── pet_cat5/                 (총 45장)
├── scene_waterfall/          (총 27장)
├── transport_tank/           (총 21장)
└── wearable_jacket1/         (총 15장)
```

---

## 5. 학습 및 인퍼런스 연계 계획

1. **SD3.5 LoRA 파인튜닝 연계**:
   - `augmentation/` 내 생성된 이미지와 `metadata.jsonl` (또는 `.txt` 페어)을 직접 Diffusers/Hugging Face LoRA 학습 파이프라인의 데이터셋 경로로 지정.
2. **RF-Inversion Multi-reference Latent Averaging**:
   - 배경 제거된 `_nobg.png`와 원본 `_std.png`를 조합하여 인버전 시 불필요한 배경 잡음을 억제하고 정체성 벡터 추출 정밀도 향상.
3. **정량/정성 평가 검증**:
   - CustomConcept101 10개 평가 프롬프트 셋에 대해 증강 데이터셋 학습 모델의 CLIP-I (서브젝트 유사도) 및 CLIP-T (텍스트 충실도) 벤치마크 측정.

---

## 6. 시각화 검증 대시보드 ([`dataset_viewer.html`](file:///content/project-3/dataset_viewer.html))

### 6.1 검증 도구 개요
데이터셋(`dataset/`)과 증강 결과물(`augmentation/`)의 품질 및 프롬프트 연계성을 한눈에 정밀 점검하기 위해 **인터랙티브 웹 대시보드 뷰어([`dataset_viewer.html`](file:///content/project-3/dataset_viewer.html))**를 구축하였습니다.

- **생성 및 구동 스크립트**: [`generate_dataset_viewer.py`](file:///content/project-3/generate_dataset_viewer.py)
- **로컬 웹 서버 URL**: `http://localhost:8000/dataset_viewer.html`
- **직접 파일 경로**: [`dataset_viewer.html`](file:///content/project-3/dataset_viewer.html)

### 6.2 대시보드 뷰어 핵심 기능
1. **서브젝트 사이드바 탐색**: 10개 서브젝트(`actionfigure_2` ~ `wearable_jacket1`) 간 자유로운 탭 이동.
2. **Original vs Augmented 데이터셋 비교 탭**:
   - 원본 데이터셋 (`dataset/` - 70장)과 증강 데이터셋 (`augmentation/` - 286장)을 원클릭으로 상호 비교.
3. **평가 프롬프트(10개) 연동 뷰**:
   - `CLASS_PROMPT`가 치환된 실제 채점용 텍스트 프롬프트를 화면 상단에 상시 노출하여 서브젝트 특성과 프롬프트의 적합성 점검.
4. **이미지 메타데이터 카드 & 라이트박스 팝업**:
   - 각 이미지의 해상도(width x height), 파일 용량(KB), 포맷 정보 표시.
   - 이미지 클릭 시 고해상도 라이트박스 모달을 통해 픽셀 깨짐, 배경 제거 정확도, 찌그러짐 유무 정밀 검사 가능.

### 6.3 뷰어 실행 및 갱신 명령어
```bash
# 데이터셋/증강 결과 변경 시 대시보드 HTML 자동 갱신 및 웹 서버 구동
python generate_dataset_viewer.py --server --port 8000
```
