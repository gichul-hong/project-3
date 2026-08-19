# Subject-driven Customization 데이터 증강 계획서 (AUGMENTATION_PLAN.md)

> **프로젝트**: VERILUX (SD3.5-medium 기반 Subject-driven Customization)  
> **관련 스크립트**: [`augment_dataset.py`](file:///C:/hong/project-3/augment_dataset.py)  
> **출력 디렉터리**: [`augmentation/`](file:///C:/hong/project-3/augmentation)  
> **최종 수정일**: 2026-08-19  

---

## 1. 개요 및 목적

### 1.1 데이터셋 한계 및 과제
- **데이터 부족**: CustomConcept101 10개 서브젝트의 레퍼런스는 서브젝트당 4~15장에 불과함.
- **해상도 및 비율 불균형**: 700px ~ 9000px 초고해상도, 정사각형/직사각형 혼재.
- **배경 간섭(Background Bleeding)**: 레퍼런스 이미지의 배경(거실, 야외 등)이 Inversion 또는 학습 시 원치 않게 수용되어, 테스트 프롬프트(`"in times square"`, `"riding a motorcycle"`)와의 배경 조화 및 다양성을 저해함.

### 1.2 증강(Augmentation)의 3대 목표
1. **정체성 유지 (CLIP-I 극대화)**: 레퍼런스 객체의 고유 픽셀 디테일과 체형/재질 정체성을 손상시키지 않고 보존.
2. **프롬프트 준수 (CLIP-T 향상)**: 레퍼런스 원본 배경을 격리하여 신규 프롬프트 배경 생성이 자유롭게 개입하도록 유도.
3. **정성 평가 다양성 확보 (30% 비중)**: 과적합(Overfitting)을 방지하고 다양한 각도/조명 반응성 확보.

---

## 2. 증강 수량 및 핵심 전략

### 2.1 적정 증강 규모 (Quantity Target)
- **증강 배율**: 원본 1장당 **2~3배 정밀 확장** (서브젝트당 총 12~15장 내외, 인물/동물은 최대 45장).
- **전체 규모**: 원본 70장 → **총 210장** 생성 확정.

| 구분 | 레퍼런스 부족 서브젝트 (4~5장) | 레퍼런스 풍부 서브젝트 (9~15장) |
|---|---|---|
| **해당 서브젝트** | `furniture_sofa2`, `decoritems_woodenpot`, `wearable_jacket1`, `luggage_backpack1` | `person_3`, `pet_cat5`, `scene_waterfall` |
| **증강 목표** | 3배 확장 (서브젝트당 **12~15장**) | 2~3배 확정 (서브젝트당 **27~45장**) |
| **목적** | 적은 데이터로 인한 과적합 방지 | 다양한 앵글 및 표현력 극대화 |

---

### 2.2 기법별 증강 전략 (Technique & Rationale)

#### 1) 📐 Center Fit & Padding (512x512) [필수]
- **원리**: 종횡비를 엄격히 유지한 채 512x512 정사각형 캔버스 중앙에 배치 및 패딩.
- **효과**: 이미지 강제 압축으로 인한 객체 찌그러짐 방지, SD3.5 입력 표준화.

#### 2) ↔️ Horizontal Flip (좌우 반전) [필수]
- **원리**: 좌우 대칭 반전 이미지 생성 (`_flip.png`).
- **효과**: 좌/우 한쪽 앵글에 치우친 정체성 학습 극복, 포즈 유연성 강화.

#### 3) 💡 Light Contrast Adjustment (1.1x 미세 대비) [권장]
- **원리**: 픽셀 정체성을 훼손하지 않는 수준(1.1x)에서의 미세 대비 조정 (`_contrast.png`).
- **효과**: 다양한 조명 및 배경 합성 시 반응성 향상.

#### 4) ✂️ Background Removal (`rembg` / SAM 전처리) [옵션/권장]
- **원리**: AI 배경 제거 모델을 통해 순수 객체(Foreground)만 흰색/투명 배경으로 분리 (`_nobg.png`).
- **효과**: Inversion/LoRA 학습 시 원본 배경 정보 얽힘(Entanglement) 차단 → CLIP-T 및 정성 다양성 급상승.
- **특이사항**: 자연 풍경 서브젝트(`scene_waterfall`)는 배경 자체가 객체이므로 배경 제거 대상에서 제외.

#### 🚫 제외된 기법 (Excluded Methods)
- **Vertical Flip (상하 반전)**: 사물/사람/동물이 뒤집히면 부자연스러운 Latent 형성.
- **Extreme Rotation & Heavy Noise**: 객체 세부 디테일(로고, 이목구비)을 훼손하여 CLIP-I 점수 하락 유발.
- **생성형 Img2Img 증강**: 생성 과정에서 디테일 변형 위험이 있으므로 전처리 중심 증강 우선 채택.

---

## 3. 서브젝트별 증강 현황 (210장)

| 서브젝트 폴더명 | class prompt | 서브젝트 특성 | 원본 수 | 증강 후 수 | 세부 증강 기법 적용 규칙 |
|---|---|---|---|---|---|
| `actionfigure_2` | action figure | 사물 / 캐릭터 (PNG) | 6장 | **18장** | CenterFit, Flip, Contrast, (rembg) |
| `decoritems_woodenpot` | wooden pot | 사물 / 화분 (PNG) | 4장 | **12장** | CenterFit, Flip, Contrast, (rembg) |
| `furniture_sofa2` | sofa | 가구 / 소파 (PNG) | 4장 | **12장** | CenterFit, Flip, Contrast, (rembg) |
| `instrument_music2` | guitar | 악기 / 기타 (PNG) | 6장 | **18장** | CenterFit, Flip, Contrast, (rembg) |
| `luggage_backpack1` | backpack | 사물 / 가방 (JPG) | 5장 | **15장** | CenterFit, Flip, Contrast, (rembg) |
| `person_3` | person | 인물 / 사람 (JPG/PNG) | 15장 | **45장** | CenterFit, Flip, Contrast, (rembg) |
| `pet_cat5` | cat | 동물 / 고양이 (JPG) | 9장 | **27장** | CenterFit, Flip, Contrast, (rembg) |
| `scene_waterfall` | waterfall | 자연 풍경 (JPG) | 9장 | **27장** | CenterFit, Flip, Contrast **(rembg 예외)** |
| `transport_tank` | tank | 탈것 / 탱크 (JPG) | 7장 | **21장** | CenterFit, Flip, Contrast, (rembg) |
| `wearable_jacket1` | jacket | 의류 / 재킷 (PNG) | 5장 | **15장** | CenterFit, Flip, Contrast, (rembg) |
| **합계** | - | - | **70장** | **총 210장** | - |

---

## 4. 실행 방법 및 파이프라인 가이드

### 4.1 스크립트 위치 및 실행 명령어
증강 파이프라인 코드: [`augment_dataset.py`](file:///C:/hong/project-3/augment_dataset.py)

```bash
# 1) 전체 10개 서브젝트 일괄 증강
python augment_dataset.py --dataset ./dataset --output ./augmentation --concept all

# 2) 특정 서브젝트만 선택 실행 (예: actionfigure_2)
python augment_dataset.py --dataset ./dataset --output ./augmentation --concept actionfigure_2

# 3) rembg 기반 배경 제거 증강 포함 실행 (pip install rembg 사전 수행)
python augment_dataset.py --dataset ./dataset --output ./augmentation --concept all --use_rembg
```

### 4.2 출력 디렉터리 구조
```text
augmentation/
├── actionfigure_2/
│   ├── 00_0_std.png        # 512x512 중앙 패딩 표준
│   ├── 00_0_flip.png       # 좌우 반전
│   ├── 00_0_contrast.png   # 1.1x 대비 조정
│   └── ... (총 18장)
├── decoritems_woodenpot/   (총 12장)
├── furniture_sofa2/        (총 12장)
├── instrument_music2/      (총 18장)
├── luggage_backpack1/      (총 15장)
├── person_3/               (총 45장)
├── pet_cat5/               (총 27장)
├── scene_waterfall/        (총 27장)
├── transport_tank/         (총 21장)
└── wearable_jacket1/       (총 15장)
```

---

## 5. 프로젝트 단계별 연계 계획

1. **Phase 1 (Baseline 확립)**:
   - 샘플 서브젝트(`actionfigure_2`, `pet_cat5`)의 원본 대비 `augmentation/` 데이터 입력 시 Latent Inversion 품질 및 CLIP-I 점수 비교.
2. **Phase 2 (정량 점수 최대화)**:
   - 배경이 제거/정리된 `augmentation/` 데이터셋을 활용하여 RF-Inversion Multi-reference Latent Averaging 수행.
3. **Phase 3 (아이디어 40% 반영)**:
   - `augmentation/` 데이터를 입력으로 SD3.5 경량 LoRA/Textual Inversion 학습 진행 후 Baseline(Inversion 단독)과의 CLIP-I/CLIP-T 점수비교 발표 자료 구성.

---

## 6. 데이터셋 및 증강 결과 시각화 검증 도구 (Dataset Viewer Dashboard)

### 6.1 검증 도구 개요
데이터셋(`dataset/`)과 증강 결과물(`augmentation/`)의 품질 및 프롬프트 연계성을 한눈에 정밀 점검하기 위해 **인터랙티브 웹 대시보드 뷰어([`dataset_viewer.html`](file:///C:/hong/project-3/dataset_viewer.html))**를 구축하였습니다.

- **생성 및 구동 스크립트**: [`generate_dataset_viewer.py`](file:///C:/hong/project-3/generate_dataset_viewer.py)
- **로컬 웹 서버 URL**: `http://localhost:8000/dataset_viewer.html`
- **직접 파일 경로**: [`dataset_viewer.html`](file:///C:/hong/project-3/dataset_viewer.html)

### 6.2 대시보드 뷰어 핵심 기능
1. **서브젝트 사이드바 탐색**: 10개 서브젝트(`actionfigure_2` ~ `wearable_jacket1`) 간 자유로운 탭 이동.
2. **Original vs Augmented 데이터셋 비교 탭**:
   - 원본 데이터셋 (`dataset/` - 70장)과 증강 데이터셋 (`augmentation/` - 210장)을 원클릭으로 상호 비교.
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
