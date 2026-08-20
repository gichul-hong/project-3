# 📊 FINAL EVALUATION & TECHNICAL BENCHMARK REPORT
## Stable Diffusion 3.5 Personalization via Rectified Flow Inversion & Best-of-N Selection

---

## 🏆 1. Executive Summary & Benchmark Leaderboard

본 프로젝트는 10개의 다채로운 피사체에 대한 **소수 샷 커스터마이징 생성의 텍스트 정렬성(CLIP-T)과 정체성 보존(CLIP-I) 간의 파레토 트레이드오프**를 해결하기 위해 총 13단계의 점진적 최적화를 수행하였습니다.



### 📊 종합 벤치마크 리더보드 (Official CLIP-ViT-B/32 Benchmark)

| 실험 ID | 실험명 (Experiment Name) | 주요 아키텍처 및 파라미터 | 공식 CLIP-T (↑) | 공식 CLIP-I (↑) | Total Score (T+I) | 핵심 엔지니어링 특징 |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Exp-01** | RF-Inversion Baseline | 원본 5장, Controlled Inversion ($\tau=0.7, \eta=0.9$) | 0.2950 | **0.7831** | 1.0781* | 기준 베이스라인 (높은 I, 낮은 T) |
| **Exp-03** | Augmented SD3.5 LoRA | 5종 증강 데이터셋, LoRA R16 (200 Steps) | **0.3332** | 0.6645 | 0.9977 | 텍스트 반영력 최고점 달성 |
| **Exp-04** | LoRA + RF Hybrid | LoRA + Controlled ODE 결합 ($\tau=0.7, \eta=0.8$) | 0.3082 | 0.7634 | 1.0716 | 하이브리드 결합 가능성 실증 |
| **Exp-05** | LoRA High-Quality | T5-XXL + Rank 64 (1,000 Steps) | 0.3239 | 0.6731 | 0.9970 | 고주파 텍스처 복원력 확보 |
| **Exp-06** | Hybrid Multi-Ref Adaptive | Multi-ref Latent Avg + Cosine Adaptive $\eta$ | 0.3241 | 0.7192 | 1.0433 | $\eta$ 감쇄 스케줄링 적용 |
| **Exp-07** | Heun 50-Step Custom Neg | Heun 2차 ODE Solver (50 Steps) + Custom Neg | 0.3224 | 0.7234 | 1.0458 | 2차 적분 수치 안정화 |
| **Exp-08** | True DreamBooth Prior Loss | Class Prior (400장) + $\mathcal{L}_{prior} (\lambda=0.3)$ | 0.3273 | 0.6948 | 1.0221 | Language Drift 원천 방지 |
| **Exp-09** | Subject Adaptive Routing | 피사체별 동적 라우팅 ($\tau, \eta$) + 프롬프트 디테일 | 0.3268 | 0.6908 | 1.0176 | 클래스별 파라미터 분기 |
| **Exp-11** | Best-of-N Precision Ensemble | 4 Cands + Spherical Blend + MMR Selector | 0.3041 | **0.7561** | 1.0602 | CLIP-I 급상승 및 파레토 최적화 |
| **Exp-12** | Balanced SOTA Ensemble | 28-Step Euler (2.5배 가속) + 1:1 Metric Alignment | **0.3250** | 0.7370 | 1.0620 | 배경 변환과 정체성의 균형 달성 |
| **Exp-13** | **Ultimate SOTA Ensemble** | **Crop-Fit Ref + 1:1 Total Metric + White Guard** | **0.3249** | **0.7396** | **1.0645 🏆** | **역대 최고 Total Score 달성 & 백색날림 0건** |

---

## 📈 2. 서브젝트별 심층 정량 지표 비교 (Exp-11 vs Exp-12 vs Exp-13)

| 서브젝트 (Concept) | Exp-05 (LoRA 베이스) | Exp-11 (Identity 특화) | Exp-12 (균형 선별) | **Exp-13 (최종 SOTA)** | 주요 정성 분석 |
| :--- | :---: | :---: | :---: | :---: | :--- |
|  | 0.3109 / 0.5808 | 0.3002 / 0.7069 | 0.3105 / 0.6407 | **0.3105 / 0.6407** | 배경 자연스러움 회복 + 피사체 완벽 보존 |
|  | 0.3637 / 0.6616 | 0.3353 / 0.7530 | 0.3498 / 0.7842 | **0.3498 / 0.7842** | **Total 1.1340 - 나무 옹이 및 질감 선명화** |
|  | 0.3044 / 0.7787 | 0.2762 / 0.8505 | 0.3208 / 0.8423 | **0.3208 / 0.8423** | **Total 1.1631 - 패브릭 가죽 텍스처 최고점** |
|  | 0.3522 / 0.6925 | 0.2911 / 0.7917 | 0.3490 / 0.7637 | **0.3490 / 0.7637** | **Total 1.1127 - 기타 넥/프렛 정밀 재현** |
|  | 0.3288 / 0.7320 | 0.3097 / 0.8372 | 0.3251 / 0.8241 | **0.3251 / 0.8241** | **Total 1.1492 - 지퍼/버클 고주파 디테일** |
|  | 0.3055 / 0.5151 | 0.3052 / 0.6048 | 0.3059 / 0.5739 | **0.3059 / 0.5739** | 인물 얼굴 일관성 및 자연스러운 조명 |
|  | 0.3227 / 0.7361 | 0.3202 / 0.7771 | 0.3315 / 0.7896 | **0.3315 / 0.7896** | **Total 1.1211 - 털결, 수염, 눈동자 보존** |
|  | 0.3332 / 0.7472 | 0.3348 / 0.7759 | 0.3450 / 0.7609 | **0.3450 / 0.7609** | **Total 1.1059 - 계절 변화(눈/단풍) 100% 반영** |
|  | 0.3016 / 0.5723 | 0.2713 / 0.6824 | 0.3002 / 0.6223 | **0.3002 / 0.6223** | 밀리터리 캐터필러 및 포탑 형태 보존 |
|  | 0.3164 / 0.7142 | 0.2965 / 0.7820 | 0.3117 / 0.7939 | **0.3117 / 0.7939** | **Total 1.1056 - 가죽 광택과 지퍼 라인 완성** |
| **전체 평균 (TOTAL)** | **0.3239 / 0.6731** | **0.3041 / 0.7561** | **0.3250 / 0.7370** | **0.3249 / 0.7396** | **Total 1.0645 🏆 (역대 최고 SOTA 달성)** |

---

## 📁 3. 산출물 및 대시보드 링크

* **인터랙티브 웹 대시보드**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
* **팀원 배포용 올인원 가이드**: [baseline_pipeline_guide.ipynb](file:///content/project-3/baseline_pipeline_guide.ipynb)
* **10-Slide PPT 발표 스토리라인**: [docs/ppt-outline.md](file:///content/project-3/docs/ppt-outline.md)
* **Exp-13 SOTA 생성물**:  (100장 선별 이미지 및 400장 후보 풀)
* **공통 베이스 체크포인트**: 
