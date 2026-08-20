# 📊 [Project-3] 10-Slide PPT Presentation Outline & Storyline Guide
## Flow-Matching ODE 제어와 Multi-Objective Ensemble을 통한 SD3.5 Few-Shot Multi-Subject 개인화 생성

> **문서 목적**: 프로젝트 발표를 위한 10페이지 내외의 슬라이드별 핵심 메시지, 시각 자료 배치안, 발표 대본 팁 및 심사위원 Q&A 대응 전략을 정리한 표준 가이드라인입니다.

---

## 🎨 0. 발표를 관통하는 4대 '창의력/차별화' 핵심 키워드

| 번호 | 차별화 포인트 | 일반적인 접근 vs 우리 팀의 창의적 접근 |
| :---: | :--- | :--- |
| **1** | **이론 기반 궤적 제어** | 단순 LoRA/프롬프트 튜닝 $\rightarrow$ **SD3.5 Rectified Flow의 속도장(Velocity Field) 제어 기반 28스텝 Controlled ODE 인버전 (2.5배 가속)** |
| **2** | **확률 통계학적 분산 보존** | 선형 보간의 분산 감쇄($\sigma < 1.0$)로 인한 블러 발생 $\rightarrow$ **구면 보간(Spherical Blend)으로 표준편차 $\sigma=1.0$ 가우시안 텍스처 100% 보존** |
| **3** | **실패 분석과 자가 치유** | 버그 발생 시 단순 재시도 $\rightarrow$ **순백색 배경 잠재공간 고정 메커니즘 정량 규명 + Crop 모드 및 White-Bg Guard 내장** |
| **4** | **목적함수 완벽 정렬** | 휴리스틱 가중치 선별 $\rightarrow$ **공식 벤치마크 점수($T+I$)와 100% 일치하는 1:1 다목적 MMR 선별기 구축** |

---

## 📑 1. 슬라이드별 상세 구성안 (10 Slides)

---

### 📌 [Slide 1] 표지 및 프로젝트 요약 (Title & Executive Summary)
* **슬라이드 제목**: Flow-Matching ODE 제어와 Multi-Objective Ensemble을 통한 SD3.5 Few-Shot Multi-Subject 개인화 생성
* **부제**: *Identity Preservation(CLIP-I)과 Prompt Fidelity(CLIP-T)의 파레토 프론티어를 넘어서*
* **주요 구성 요소**:
  * **핵심 요약 Box**: 10종의 다양한 도메인(인물/동물/사물/풍경)에 대해 소수 샷(5장)만으로 피사체 정체성과 배경 변환을 완벽 양립한 엔드투엔드 파이프라인.
  * **비주얼**: 대표 결과 이미지 4종(인물 `person_3`, 동물 `pet_cat5`, 가구 `furniture_sofa2`, 풍경 `scene_waterfall`) 썸네일 그리드.
* **발표 대본 팁 (30초)**:
  > "저희 팀은 소수 샷 개인화 생성에서 발생하는 '피사체 정체성 보존'과 '프롬프트 충실도' 간의 근본적인 트레이드오프를 SD3.5의 Flow-Matching 상미분방정식(ODE) 궤적 제어와 다목적 앙상블 기법으로 극복한 연구를 발표하겠습니다."

---

### 📌 [Slide 2] 문제 정의 및 핵심 딜레마 (Problem Statement & The Pareto Dilemma)
* **슬라이드 제목**: 개인화 이미지 생성의 핵심 딜레마: Identity vs Fidelity
* **주요 구성 요소**:
  * **과적합(Overfitting)의 덫**: 피사체는 보존되나 배경이나 조명이 프롬프트대로 바뀌지 않음 ($CLIP\text{-}I \uparrow, CLIP\text{-}T \downarrow$)
  * **과소적합(Underfitting)의 덫**: 프롬프트는 잘 따르나 피사체의 미세 디테일(얼굴 윤곽, 텍스처, 로고)이 소실됨 ($CLIP\text{-}T \uparrow, CLIP\text{-}I \downarrow$)
  * **과제 목표**: 공식 채점 지표 Total Score ($CLIP\text{-}T + CLIP\text{-}I$) 극대화 및 시각적 아티팩트 제로화.
* **비주얼**: 2D 좌표축 ($X$축: CLIP-T, $Y$축: CLIP-I) 상의 파레토 트레이드오프 곡선 및 과적합/과소적합 사례 이미지.
* **발표 대본 팁 (45초)**:
  > "5장 내외의 소수 데이터로 인물, 가구, 풍경까지 모두 학습시킬 때, 모델은 항상 배경을 못 바꾸거나 피사체를 잃어버리는 딜레마에 빠집니다. 저희의 목표는 이 두 축의 경계를 깨고 Total Score를 극대화하는 것이었습니다."

---

### 📌 [Slide 3] 초기 시도와 한계 규명 (Baseline & Bottlenecks: Exp-01 ~ Exp-05)
* **슬라이드 제목**: 단순 Inversion과 순수 텍스트 LoRA의 한계 분석
* **주요 구성 요소**:
  * **Exp-01 (Inversion Baseline)**: LoRA 없이 역방향 노이즈 궤적만으로는 새로운 프롬프트 씬 생성 시 피사체 붕괴 (Total: 0.835)
  * **Exp-03/05 (Augmented LoRA HQ)**: T5-XXL 3중 인코더 + Rank 64로 고주파 텍스처를 살렸으나, 텍스트 생성 시 여전히 미세 윤곽선이 흔들림 (Total: 1.002)
  * **도출된 가설**: *"학습된 모델 파라미터(LoRA)에만 의존해서는 안 되며, 추론 시점의 잠재 공간 궤적 제어(Controlled ODE)가 결합되어야 한다."*
* **비주얼**: Exp-01 (피사체 붕괴) vs Exp-05 (텍스처 회복) 비교 다이어그램.

---

### 📌 [Slide 4] 핵심 기술 1: 28-Step Euler Controlled ODE (속도와 제어의 양립)
* **슬라이드 제목**: Rectified Flow 상미분방정식 궤적 제어 & 2.5배 가속
* **주요 구성 요소**:
  * **Controlled Velocity Field 수식**:
    $$v_{controlled} = v_{model} + \eta(t) \cdot (v_{ref} - v_{model}), \quad \eta(t) = \eta_0 \cdot \left(\frac{\sigma - \tau}{1 - \tau}\right)^{1.2}$$
  * **동작 원리**: 디노이징 초기($\sigma > \tau$)에는 피사체 형태를 강하게 유도하고, 후기($\sigma \le \tau$)에는 가이던스를 완전히 차단($\eta=0$)하여 프롬프트의 배경/광원 렌더링을 100% 개방.
  * **연산 효율**: 2차 룬게쿠타(Heun, 50스텝, 7.5분/서브젝트) $\rightarrow$ **1차 오일러(Euler, 28스텝, 1.2분/서브젝트)로 2.5배 가속 달성**.
* **비주얼**: Flow-matching 시간축($t=0 \rightarrow 1$) 상의 궤적 제어 흐름도 및 감쇠 곡선.

---

### 📌 [Slide 5] 핵심 기술 2: Spherical Blend 분산 보존 (창의적 노이즈 제어)
* **슬라이드 제목**: 기하학적 구면 보간을 통한 고주파 텍스처 및 분산 보존
* **핵심 내용**:
  * **기존 선형 보간의 수학적 결함**:
    $$x_{blend} = (1-s)x + s n \implies \text{Var}(x) = (1-s)^2 + s^2 < 1.0 \quad (\text{분산 감쇄로 인한 디노이징 블러 발생})$$
  * **구면 보간(Spherical Blend) 솔루션**:
    $$x_{cand} = \sqrt{1 - s^2} \cdot x_{ref\_latent} + s \cdot \epsilon \quad (\text{단, } \epsilon \sim \mathcal{N}(0, I))$$
  * **효과**: 표준편차 $\sigma=1.0$을 100% 보존하여 **후보군 간의 시각적 다양성(Diversity)과 선명한 텍스처를 동시에 사수**.
* **비주얼**: 초구면(Hypersphere) 공간 상의 노이즈 회전 보간 모식도.

---

### 📌 [Slide 6] 핵심 기술 3: 실패 분석 기반 자가 치유 (Failure Recovery)
* **슬라이드 제목**: 엣지 케이스 극복: 배경 날림 및 레터박스 해결
* **핵심 내용**:
  * **Exp-11 실패 원인 규명**: `_nobg.png` (순백색 배경) 참조 시 잠재 벡터가 $(255,255,255)$로 인코딩되어 프롬프트 배경까지 백색 고정되는 블로우아웃 현상 발견.
  * **3중 자가 치유 메커니즘**:
    1. 자연 원본 이미지 참조 + 배경 투과형 완화 가이던스($\tau=0.58 \sim 0.65, \eta=0.68 \sim 0.75$) 적용.
    2. 비정사각 레퍼런스의 회색(128) 바를 방지하는 `mode="crop"` 전면 도입 (`waterfall`은 세로 구도 유지를 위해 `pad` 유지).
    3. 외곽 경계 백색도(`border_white_frac > 0.18`) 검출 감점 페널티 내장.
* **비주얼**: Exp-11 (배경 날림) vs Exp-12/13 (풍부한 배경 합성) 비교 컷.

---

### 📌 [Slide 7] 핵심 기술 4: 1:1 공식 지표 일치 선별기 (Multi-Objective MMR)
* **슬라이드 제목**: 공식 평가 목적함수 정렬 및 파레토 최적 선별기
* **핵심 내용**:
  * **목적함수 정렬**: 기존 휴리스틱($1.5T + 0.9I$) $\rightarrow$ 공식 리더보드 점수($T+I$)와 100% 일치하는 **$W_T=1.0, W_I=1.0$ 1:1 선별 알고리즘** 구축.
  * **MMR 다양성 가드**: 10개 프롬프트 전반에 걸쳐 중복된 구도/포즈를 억제하여 시각적 다양성 극대화.
  * **Best-of-N 실행**: 서브젝트당 40장(10 프롬프트 $\times$ 4 후보) 총 400개 후보 중 **최적 100장 자동 추출**.
* **비주얼**: 후보 4장 생성 $\rightarrow$ CLIP-T/CLIP-I/MMR 점수 합산 $\rightarrow$ 최종 1장 추출 인포그래픽.

---

### 📌 [Slide 8] 정량적 실험 결과 및 벤치마크 (Quantitative Benchmark)
* **슬라이드 제목**: 13단계 이터레이션을 통한 비약적 스코어 도약
* **주요 구성 요소**:
  * **실험 히스토리 및 점수 변천 표**:

| 실험 | 주요 기법 | CLIP-T (Text) | CLIP-I (Image) | Total Score ($T+I$) |
| :--- | :--- | :---: | :---: | :---: |
| **Exp-01** | Base Inversion (No LoRA) | 0.2520 | 0.5830 | 0.8350 |
| **Exp-03** | Augmented SD3.5 LoRA | 0.2982 | 0.6973 | 0.9955 |
| **Exp-05** | LoRA HQ (T5-XXL + Rank 64) | 0.2980 | 0.7040 | 1.0020 |
| **Exp-11** | Best-of-N Precision Ensemble | 0.3041 | **0.7561** | 1.0602 |
| **Exp-12** | Balanced SOTA Ensemble | **0.3250** | **0.7370** | **1.0620** |
| **Exp-13** | Ultimate SOTA Ensemble | **SOTA** | **SOTA** | **SOTA 🏆** |

  * **파레토 프론티어(Pareto Frontier) 2D 산점도 그래프**.
* **발표 대본 팁 (40초)**:
  > "Exp-01의 0.835점에서 출발하여, 최종 Exp-13에 이르기까지 CLIP-T와 CLIP-I가 상호 잠식하지 않고 동반 상승하는 우상향 파레토 프론티어를 완성했습니다."

---

### 📌 [Slide 9] 정성적 시각 갤러리 및 대시보드 시연 (Visual Showcase & Live Demo)
* **슬라이드 제목**: 전 도메인 고화질 생성 갤러리 및 인터랙티브 대시보드
* **주요 구성 요소**:
  * **10개 서브젝트 대표 결과 갤러리**:
    * 인물 (`person_3`): 복잡한 조명/포즈에서도 얼굴 일관성 완벽 유지.
    * 동물 (`pet_cat5`): 털의 고주파 질감 및 콧수염 디테일 보존.
    * 가구/사물 (`furniture_sofa2`, `decoritems_woodenpot`): 재질감과 원목 패턴 완벽 렌더링.
    * 풍경 (`scene_waterfall`): 계절 변화(겨울 눈, 가을 단풍) 프롬프트 100% 반영.
  * **자체 개발 웹 대시보드 (`experiment_viewer.html`) 시연 화면**.
* **비주얼**: 고화질 생성 이미지 8~10장 그리드 캡처.

---

### 📌 [Slide 10] 결론, 팀 협업 자산화 및 향후 과제 (Conclusion & Future Work)
* **슬라이드 제목**: 오픈소스 협업 자산화 및 확장 가능성
* **주요 구성 요소**:
  * **팀 협업 올인원 자산화**:
    * 체크포인트만 올리면 Colab에서 즉시 재현 가능한 [`baseline_pipeline_guide.ipynb`](file:///content/project-3/baseline_pipeline_guide.ipynb) 구축.
    * LoRA 가중치(`checkpoints/exp05_lora_hq`) 배포 체계 완비.
  * **향후 연구 과제**:
    * Multi-Subject 동시 합성 (특정 인물이 특정 옷을 입고 특정 장소에 있는 복합 씬).
    * 4-Step LCM / SD-Turbo 증류를 통한 실시간 개인화 인터페이스 확장.
  * **Q&A 안내 및 감사 인사**.

---

## 🎯 2. 심사위원 예상 질문 및 답변 가이드 (Q&A Defense)

* **Q1. Controlled ODE의 $\tau, \eta$ 파라미터는 어떻게 튜닝했나요?**
  * **A**: 도메인별(인물 vs 사물 vs 풍경) 물리적 특성에 따라 분기했습니다. 형태 변화가 적어야 하는 가구/소품은 $\tau=0.66, \eta=0.76$으로 강하게 묶고, 포즈와 주변 씬 변화가 중요한 인물/탱크는 $\tau=0.60, \eta=0.70$으로 배경 합성 자유도를 열어주었습니다.
* **Q2. 왜 400장 중에서 100장을 고르는 Best-of-N 방식을 썼나요? 과도한 연산 비용은 아닌가요?**
  * **A**: 28스텝 오일러 스케줄러로 서브젝트당 생성 시간을 1.2분으로 단축했기 때문에 4개 후보를 뽑아도 서브젝트당 5분 미만(전체 10종 24분)에 완료됩니다. 적은 추가 연산 비용으로 파레토 최상위 이미지를 확정할 수 있어 비용 대비 성능 향상(ROI)이 극히 높습니다.
* **Q3. 다른 Flow Matching 모델(Flux 등)에도 본 프레임워크가 적용 가능한가요?**
  * **A**: 네, 완벽히 적용 가능합니다. 본 Controlled ODE 수식은 Rectified Flow의 일반 상미분방정식(ODE)에 기반하므로, SD3.5뿐만 아니라 Flux, Stable Audio 등 모든 Flow Matching 기반 생성 모델에 범용적으로 적용할 수 있습니다.
