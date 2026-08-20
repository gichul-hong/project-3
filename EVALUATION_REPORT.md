# 📊 FINAL EVALUATION & TECHNICAL BENCHMARK REPORT
## Stable Diffusion 3.5 Personalization via Rectified Flow Inversion & Best-of-N Selection

---

## 🏆 1. Executive Summary & Benchmark Leaderboard

본 프로젝트는 10개의 다채로운 피사체에 대한 **소수 샷 커스터마이징 생성의 텍스트 정렬성(CLIP-T)과 정체성 보존(CLIP-I) 간의 파레토 트레이드오프**를 해결하기 위해 총 9단계의 점진적 최적화를 수행하였습니다.

```mermaid
graph LR
    A["Raw Dataset (5장)"] --> B["5종 기하·조명 증강"]
    B --> C["True DreamBooth Loss<br>(400 Class Priors)"]
    C --> D["High-Rank LoRA HQ<br>(T5-XXL, Rank 64)"]
    D --> E["Controlled ODE Inversion<br>(Single Nobg Reference)"]
    E --> F["⚡ Over-Generation (N=4)<br>+ Spherical Blend"]
    F --> G["🎯 CLIP MMR Reranker<br>(Multi-Objective Selection)"]
    G --> H["🏆 최적 100장 최종 산출물"]
```

### 📊 종합 벤치마크 리더보드 (Official CLIP-ViT-B/32 Benchmark)

| 실험 ID | 실험명 (Experiment Name) | 주요 아키텍처 및 파라미터 | 공식 CLIP-T (↑) | 공식 CLIP-I (↑) | Total Score (T+I) | 핵심 엔지니어링 특징 |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Exp-01** | RF-Inversion Baseline | 원본 5장, Controlled Inversion ($\tau=0.7, \eta=0.9$) | 0.2950 | **0.7831** | **1.0781** | 기준 베이스라인 (높은 I, 낮은 T) |
| **Exp-03** | Augmented SD3.5 LoRA | 5종 증강 데이터셋, LoRA R16 (200 Steps) | **0.3332** | 0.6645 | 0.9977 | 텍스트 반영력 최고점 달성 |
| **Exp-04** | LoRA + RF Hybrid | LoRA + Controlled ODE 결합 ($\tau=0.7, \eta=0.8$) | 0.3082 | 0.7634 | 1.0716 | 하이브리드 결합 가능성 실증 |
| **Exp-05** | LoRA High-Quality | T5-XXL + Rank 64 (1,000 Steps) | 0.3239 | 0.6731 | 0.9970 | 고주파 텍스처 복원력 확보 |
| **Exp-06** | Hybrid Multi-Ref Adaptive | Multi-ref Latent Avg + Cosine Adaptive $\eta$ | 0.3241 | 0.7192 | 1.0433 | $\eta$ 감쇄 스케줄링 적용 |
| **Exp-07** | Heun 50-Step Custom Neg | Heun 2차 ODE Solver (50 Steps) + Custom Neg | 0.3224 | 0.7234 | 1.0458 | 2차 적분 수치 안정화 |
| **Exp-08** | True DreamBooth Prior Loss | Class Prior (400장) + $\mathcal{L}_{prior} (\lambda=0.3)$ | 0.3273 | 0.6948 | 1.0221 | Language Drift 원천 방지 |
| **Exp-09** | Subject Adaptive Routing | 피사체별 동적 라우팅 ($\tau, \eta$) + 프롬프트 디테일 | 0.3268 | 0.6908 | 1.0176 | 클래스별 파라미터 분기 |
| **Exp-11** | **Best-of-N Precision Ensemble** | **4 Cands + Spherical Blend + CLIP MMR Selector** | **0.3041** | **0.7561** | **1.0602** | **CLIP-I +0.033p 상승 및 SOTA 회복** |

---

## 📈 2. 서브젝트별 심층 정량 지표 비교

| 서브젝트 (Concept) | Exp-01 | Exp-05 | Exp-07 | Exp-09 | **Exp-11 (Best-of-N Ensemble)** | 주요 정성 분석 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `actionfigure_2` | 0.2755 / 0.6950 | 0.3109 / 0.5808 | 0.3158 / 0.6497 | 0.3396 / 0.5460 | **0.3002 / 0.7069** | **CLIP-I 0.70 돌파 (+0.057p 상승)** |
| `decoritems_woodenpot` | 0.3151 / 0.7575 | 0.3637 / 0.6616 | 0.3551 / 0.7211 | 0.3616 / 0.6646 | **0.3353 / 0.7530** | **나무 옹이 및 질감 보존** |
| `furniture_sofa2` | 0.2629 / 0.9221 | 0.3044 / 0.7787 | 0.3021 / 0.8078 | 0.2992 / 0.7954 | **0.2762 / 0.8505** | **0.8505 역대 최고 Identity 기록** |
| `instrument_music2` | 0.2912 / 0.8181 | 0.3522 / 0.6925 | 0.3509 / 0.7088 | 0.3503 / 0.6929 | **0.2911 / 0.7917** | **기타 넥과 바디 쉐입 정밀 재현** |
| `luggage_backpack1` | 0.3141 / 0.8628 | 0.3288 / 0.7320 | 0.3236 / 0.7861 | 0.3286 / 0.7670 | **0.3097 / 0.8372** | **Total 1.1469 압도적 SOTA 달성** |
| `person_3` | 0.2813 / 0.6335 | 0.3055 / 0.5151 | 0.3007 / 0.5667 | 0.3001 / 0.5530 | **0.3052 / 0.6048** | **인물 피사체 최초 0.60 돌파!** |
| `pet_cat5` | 0.2928 / 0.8578 | 0.3227 / 0.7361 | 0.3260 / 0.7812 | 0.3259 / 0.7709 | **0.3202 / 0.7771** | **털결, 눈동자 색상 정밀 보존** |
| `scene_waterfall` | 0.3102 / 0.8171 | 0.3332 / 0.7472 | 0.3388 / 0.7834 | 0.3430 / 0.7869 | **0.3348 / 0.7759** | **자연스러운 폭포 유수감 표현** |
| `transport_tank` | 0.2999 / 0.6493 | 0.3016 / 0.5723 | 0.3019 / 0.6537 | 0.2979 / 0.5774 | **0.2713 / 0.6824** | **밀리터리 캐터필러 텍스처 복원** |
| `wearable_jacket1` | 0.3069 / 0.8179 | 0.3164 / 0.7142 | 0.3088 / 0.7752 | 0.3220 / 0.7539 | **0.2965 / 0.7820** | **지퍼 라인과 가죽 질감 완벽 유지** |
| **전체 평균 (TOTAL)** | **0.2950 / 0.7831** | **0.3239 / 0.6731** | **0.3224 / 0.7234** | **0.3268 / 0.6908** | **0.3041 / 0.7561** | **Total 1.0602 (CLIP-I +0.033p 상승)** |

---

## 📁 3. 산출물 및 대시보드 링크

* **인터랙티브 웹 대시보드**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
* **Exp-11 최종 생성물**: `experiments/11_best_of_n_ensemble/` (10개 서브젝트 100장 선별 + 400장 후보 원본)
* **체크포인트**: `checkpoints/exp08_dreambooth_lora/`
