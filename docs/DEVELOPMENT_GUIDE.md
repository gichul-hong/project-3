# VERILUX 프로젝트 개발 가이드 (Subject-driven Customization)

> 이 문서는 `docs/guidance.txt`, `docs/PDF/ProjectOverview.pdf`, `docs/PDF/Day1-1.pdf`, `docs/PDF/Day1-2.pdf`를 OCR 파싱하여 정리한 종합 가이드입니다.
> 다른 Agent(구현 담당)가 이 문서만으로 과제를 수행할 수 있도록 최대한 상세하게 작성했습니다.
> 평가 코드: `evaluation.py`, 실습 코드: `Day1_diffusers_tutorial.ipynb`, `Day1_RF_Inversion_Practice_exercise.ipynb`

---

## 1. 전반적인 내용 (개요)

### 1.1 프로젝트 목표
- **Subject-driven customization (주체 중심 커스터마이제이션)**: 주어진 데이터셋의 객체(subject)를 **유지한 채** 새로운 이미지를 생성하는 것.
- 즉, 3~15장의 레퍼런스 이미지로 특정 객체(예: 액션 피규어, 고양이, 소파 등)의 정체성을 학습/반영하여, **테스트 프롬프트 10개에 대응하는 이미지 10장**을 생성해야 한다.

### 1.2 데이터셋 (CustomConcept101)
- CustomConcept101 Dataset 중 **10개 서브젝트**만 사용.
- `dataset/` 폴더: 서브젝트별 레퍼런스 이미지 (폴더명 = 서브젝트명)
- `prompt/` 폴더: 서브젝트별 테스트 프롬프트 10개 (파일명 = 서브젝트명.txt)
- 두 폴더의 이름은 서브젝트명으로 **1:1 대응**.

### 1.3 사용할 10개 서브젝트와 class prompt, 레퍼런스 수

| 서브젝트 폴더명 | class prompt (`{}` 대체) | 레퍼런스 이미지 수 | 파일 포맷 |
|---|---|---|---|
| actionfigure_2 | action figure | 6 | png |
| decoritems_woodenpot | wooden pot | 4 | png |
| furniture_sofa2 | sofa | 4 | png |
| instrument_music2 | guitar | 6 | png |
| luggage_backpack1 | backpack | 5 | jpg |
| person_3 | person | 15 | jpg/png 혼합 |
| pet_cat5 | cat | 9 | jpg |
| scene_waterfall | waterfall | 9 | jpg |
| transport_tank | tank | 7 | jpg |
| wearable_jacket1 | jacket | 5 | png |

- class prompt는 `evaluation.py`의 `CLASS_PROMPT` 딕셔너리에 하드코딩되어 있음.
- 레퍼런스 이미지 크기는 서브젝트마다 제각각 (예: transport_tank는 ~1000px 정사각형, scene_waterfall은 4000~9000px 초고해상도, person_3은 743~2832px 혼합).

### 1.4 평가 항목 (총 100%)

| 항목 | 배점 | 설명 |
|---|---|---|
| 정량 평가 (CLIP-I & CLIP-T) | 30% | Image-to-Image (객체 정체성 유지) 및 Text-to-Image (프롬프트 준수) 점수 합계 |
| 정성 평가 | 30% | 다양성: 획일적이지 않고 다양한 이미지인가 / 레퍼런스를 그대로 복사한 수준이 심하지 않은가 |
| 아이디어 | 40% | 논문/실습 외 새로운 방법론 시도, 단순 학습량·하이퍼파라미터 튜닝 이상의 기여, 기존 기법의 창의적 응용/변형 |

- **정량 평가**: 서브젝트별 10개 프롬프트에 대해 10장 생성, `evaluation.py`로 CLIP-T/CLIP-I 두 숫자 출력.
- **CLIP 모델**: `openai/clip-vit-base-patch32` (CLIP-B/32, huggingface).
- **정성 평가**: 생성 이미지를 발표 슬라이드에 첨부해 평가받음.
- **아이디어(40%)가 가장 큰 비중**이므로 단순 baseline만 돌리는 것보다 창의적 응용/변형이 핵심.

### 1.5 제출 사항
1. **생성 이미지 (.zip)**: 서브젝트별 10장, 총 100장. 파일명은 프롬프트 순서에 맞게 `0.png` ~ `9.png`.
   - ⚠️ **순서가 프롬프트 줄 순서와 반드시 일치**해야 함. i번째 이미지가 i번째 프롬프트로 채점됨. `glob` 정렬 기준이라 `0.png ... 9.png` 형식 권장 (10.png가 되면 0 다음 10이 오는 정렬 문제 주의).
2. **발표 자료**:
   - CLIP Score: CLIP-I/CLIP-T 각각 서브젝트별 평균 + 전체 평균 = (10+1) × 2 = **22개 값**.
   - 생성 이미지 일부만 첨부 (전체 불필요).
   - 코드 별도 제출 없음.
   - **실습 코드 활용 영역 vs 본인 아이디어 반영 영역을 구분**해 발표.
   - 서브젝트별/이미지별 방법론이 다를 경우: (1) 통일했을 때 점수, (2) 개별 적용했을 때 점수를 나눠 기재.

### 1.6 하드 제약 (반드시 준수)
- **베이스 모델**: `stabilityai/stable-diffusion-3.5-medium` (SD3.5-medium) 만 사용.
  - HuggingFace에서 라이선스 동의(게이팅) 후 액세스 토큰으로 로그인 필요.
- **프롬프트는 수정 가능** (Textual Inversion/DreamBooth 등 방법마다 baseline prompt가 달라서). 단, 변경 시 발표자료에 명시. **평가 시에는 `evaluation.py`에 명시된 원본 프롬프트를 그대로 사용.**
- 평가 시 `evaluation.py`를 서브젝트 하나씩 실행.

---

## 2. 구체적인 지침 (구현 상세)

### 2.1 평가 코드 (`evaluation.py`) 동작 방식
```
python evaluation.py --dataset ./dataset --prompts ./prompt --concept actionfigure_2 --images ./generated/actionfigure_2
```
- `--dataset`: 레퍼런스 이미지 상위 폴더
- `--prompts`: 프롬프트 txt 폴더 (프로젝트 루트의 `prompt` 폴더 지정)
- `--concept`: 평가할 서브젝트 하나 (10개 중 택1)
- `--images`: 생성 이미지 10장 폴더
- 동작:
  1. `ref_paths = sorted(glob(...))`, `gen_paths = sorted(glob(...))` → **파일명 정렬 순서가 곧 채점 순서**.
  2. 프롬프트 파일을 읽어 `{}`를 class prompt로 치환 (`photo of a cat` 등).
  3. CLIP-B/32로 텍스트/이미지 임베딩 추출.
  4. **CLIP-T**: `i`번째 생성 이미지와 `i`번째 프롬프트의 cosine similarity (1:1 짝).
  5. **CLIP-I**: 모든 생성 이미지와 모든 레퍼런스 이미지 간 cosine similarity (전체 쌍의 평균).
- CLIP-I는 전체 쌍 평균이므로, 생성 이미지가 **레퍼런스 집합과 전반적으로 유사**하면 높게 나옴 (대표 이미지 1장과 비교하는 것이 아님).

### 2.2 프롬프트 (10개 서브젝트 × 10개)
- 프롬프트 파일의 `{}`가 class prompt로 치환됨. 예: `photo of a cat`, `{} in times square.` → `cat in times square.`
- 각 서브젝트별 프롬프트 전체 내용은 `prompt/<subject>.txt` 참고. 핵심 예시(actionfigure_2):
  1. `photo of a {}.`
  2. `The {}, surrounded by towering skyscrapers.`
  3. `The {} stands atop a snowy mountain peak, overlooking a vast icy landscape.`
  4. `In a lush forest stands {}.`
  5. `{} in times square.`
  6. `{} on a sandy beach, with waves in the background.`
  7. `{} riding a motorcycle.`
  8. `{} riding a flying broom.`
  9. `{} holding a futuristic energy sword.`
  10. `The {} rides a majestic dragon.`
- 나머지 서브젝트는 유사한 포맷 (장소/상황 배치). 전체 목록은 아래 "프롬프트 전체" 섹션 참고.

### 2.3 사용 가능한 방법론 (ProjectOverview p.15 명시)
- **데이터 조작/추가/증강** (manipulate/add/augment data)
- **스케줄러(Inversion 방법) 수정**
- **네트워크 학습**: Textual Inversion, LoRA, ControlNet
- **하이퍼파라미터 튜닝**
- 프롬프트 수정 (발표 시 명시 필수)

### 2.4 실습에서 배운 내용 (Day1-1, Day1-2)

#### A. Diffusers / SD3.5 기본 (Day1-1)
- **모델 로드** (T5 XXL 비활성화 + CPU offload):
  ```python
  from diffusers import StableDiffusion3Pipeline
  pipeline = StableDiffusion3Pipeline.from_pretrained(
      "stabilityai/stable-diffusion-3.5-medium",
      text_encoder_3=None, tokenizer_3=None,
      torch_dtype=torch.bfloat16,  # Colab A100 등 Ampere GPU 권장 (float16보다 수치 안정적)
  )
  pipeline.enable_model_cpu_offload()
  ```
  - `text_encoder_3=None, tokenizer_3=None` → T5 XXL(4.7B) 생략 (효율성).
  - Colab A100(Ampere GPU)에서는 `torch.bfloat16`이 `float16` 대비 언더플로우/수치 불안정성을 방지해 안정적입니다.
  - VRAM > 8GB면 red line(위 인자들) 제거 가능.
- **생성** 핵심 파라미터:
  - `prompt`, `negative_prompt`, `num_inference_steps`(기본 28), `height`, `width`, `guidance_scale`(CFG), `generator=torch.Generator().manual_seed(0)` (재현성).
- **CFG (Classifier-Free Guidance)**: 두 번 forward(`εu` 무조건, `εc` 조건), `ε = εu + s(εc - εu)`. s=0 무조건 생성, s≈1 조건 생성, s>1 강한 가이던스. 기본 7.0.
- **negative_prompt**로 원치 않는 특성 회피 가능 (예: `"bad quality, low-resolution, distorted"`).
- **스케줄러 교체**: `FlowMatchHeunDiscreteScheduler.from_config(pipeline.scheduler.config)` 등.
  - Euler solver: 직선 이동 (1차), Heun solver: 이산화 오차 보정 (2차).
- **SD3는 Flow Matching (Rectified Flow) 기반**: `x_t = t·x1 + (1-t)·x0`, velocity `v(x,t)=x1-x0` 학습. MMDiT 백본, 3개 텍스트 인코더(CLIP-G/14, CLIP-L/14, T5-XXL).

#### B. Inversion (Day1-2) — 핵심 실습
- **목표**: 이미지 → latent 역변환, 이를 조작해 편집/생성.
- **VAE latent 인코딩** (이미지 → SD3 latent):
  ```python
  @torch.no_grad()
  def encode_image_to_sd3_latent(pipe, image, seed=0):
      device = pipe._execution_device
      image_tensor = pipe.image_processor.preprocess(image).to(device=device, dtype=pipe.vae.dtype)
      posterior = pipe.vae.encode(image_tensor).latent_dist
      raw_latent = posterior.sample(generator=torch.Generator(device=device).manual_seed(seed))
      shift_factor = pipe.vae.config.shift_factor
      scaling_factor = pipe.vae.config.scaling_factor
      return (raw_latent - shift_factor) * scaling_factor
  ```
- **Euler Inversion** (시간 역전, data→noise):
  ```python
  class EulerInversion(FlowMatchEulerDiscreteScheduler):
      def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
          super().set_timesteps(...)
          self.timesteps = torch.flip(self.timesteps, dims=(0,))
          self.sigmas = torch.flip(self.sigmas, dims=(0,))
  ```
  - Inversion 시 `guidance_scale=1.0`, `output_type="latent"`, `latents=image_latent`.
- **Heun Inversion**: sigma를 뒤집고 terminal sigma를 `1.0`으로 설정 (2차 정확도).
- **RF-Inversion** (Rout et al., ICLR 2025) — Controlled ODE:
  - 역변환 시 atypical latent 문제 해결 위해 velocity를 보간.
  - Inversion (sampled prior `z ~ P_noise` 방향): `v(x_t|z) = (z - x_t)/(1-t)`, 기본 `tau=0.0, eta=0.5`, null-text inversion (`prompt=""`).
  - Generation (reference `x0 ~ P_data` 방향): `v(x_t|x0) = (x_t - x0)/t`, 기본 `tau=0.7, eta=0.9`.
  - 공식:
    - Inversion: `V(x;t) = v_θ(x_t;t) + γ(v(x_t|z) - v_θ(x_t;t))`, γ=0.5
    - Generation: `V(x;t) = v_θ(x_t;t) + η(v(x_t|x0) - v_θ(x_t;t))`, η=0.9 for t>0.7 else 0
  - 구현: `ControlledODE`(생성), `ControlledODEInversion`(역변환) 클래스가 `FlowMatchEulerDiscreteScheduler` 상속.
  - `controller` 구현 (notebook TODO):
    - ControlledODE.controller: `return (sample - reference) / sigma.clamp_min(1e-6)` (즉 `(x_t - x0)/t` 형태, sigma≈t)
    - ControlledODEInversion.controller: `return (reference - sample) / (1.0 - sigma).clamp_min(1e-6)` (즉 `(z - x_t)/(1-t)`)
  - step 함수에서 `controlled_velocity = model_output + eta*(conditional_velocity - model_output)`.

### 2.5 환경 (하드웨어)
- **맥미니 (16GB RAM)**: 로컬 테스트/경량 작업용.
- **Google Colab A100**: 본격 학습·생성용. T4/A100 GPU 런타임.
- 로컬 conda env `pjt-3` (Python 3.12) 존재.
- Windows 머신 기준: base miniforge Python(`C:\ProgramData\miniforge3\python.exe`)에 PyMuPDF/PIL 있음 (PDF OCR에 사용).

---

## 3. 작업 플랜 (Work Plan)

### 3.1 단계별 로드맵

> 💡 **[핵심 진행 전략] 샘플 검증 후 횡전개 (Fast Prototyping & Rollout)**  
> 10개 서브젝트(100장) 전체를 대상으로 매번 실험을 반복하는 것은 계산 자원 및 시간 효율성이 떨어집니다.  
> **대표 샘플 서브젝트 1~2개**를 선정하여 파이프라인 구축 및 아이디어를 빠르게 검증(Fast Prototyping)한 뒤, 점수 향상이 확인되면 **전체 10개 서브젝트로 횡전개(Rollout)**하는 전략을 취합니다.
> - **추천 샘플 서브젝트 (2종)**:
>   1. `actionfigure_2` (사물/단일 객체 대표, 6장 png)
>   2. `pet_cat5` 또는 `person_3` (생물/복합 인물/다수 레퍼런스 대표, 9~15장)

**Phase 0 — 준비**
1. HuggingFace 계정 가입 + `stabilityai/stable-diffusion-3.5-medium` 라이선스 동의(게이팅).
2. 액세스 토큰: 이미 **`C:\hong\project-3\.env`** 에 `HF_TOKEN=hf_...` 로 저장되어 있음 (로컬).
   - 로컬 실행 시 `.env`에서 `HF_TOKEN`을 로드해 사용 (예: `python-dotenv`의 `load_dotenv()` 또는 `os.environ` 직접 주입).
   - Colab에서는 별도 환경이라 `.env`가 없으므로, Colab Secret에 `HF_TOKEN`을 등록하거나 `huggingface_hub.login()`으로 수동 로그인 필요.
3. Colab A100 런타임 선택. `dataset/`, `prompt/`, `evaluation.py` 업로드.
4. 환경 구축: `torch`, `diffusers`, `transformers`, `accelerate`, `safetensors`, `PIL`, `numpy`.
5. `evaluation.py` 로컬 실행 확인 (CPU로도 CLIP은 동작).

**Phase 1 — Baseline 확립 (샘플 1~2개 우선 적용 후 횡전개)**
1. `Day1_RF_Inversion_Practice_exercise.ipynb`의 inversion 파이프라인을 **샘플 서브젝트 (`actionfigure_2`, `pet_cat5`)**에 우선 적용.
2. 서브젝트당 레퍼런스 이미지 1장(또는 대표 1장)을 latent로 인코딩 → Euler/RF-Inversion → 10개 프롬프트로 생성 (`0.png ~ 9.png` 저장).
3. `evaluation.py`로 샘플 서브젝트의 CLIP-I/CLIP-T Baseline 점수 확인.
4. 샘플에서 정상 작동 확인 후 **전체 10개 서브젝트로 횡전개하여 Baseline 점수 확립**.
5. **주의**: 생성 이미지 순서 = 프롬프트 순서 (`0.png` = 1번 프롬프트).

**Phase 2 — 개선 (샘플 검증 → 10개 서브젝트 횡전개)**
- **정체성 유지(CLIP-I) 개선 방법**:
  - Textual Inversion / LoRA (SD3.5에서 가능한지 확인 필요 — SD3.5는 diffusers에서 LoRA/TI 지원 여부 검증).
  - RF-Inversion의 `eta/tau` 튜닝, reference latent를 "레퍼런스 집합의 평균/대표"로 선정.
  - 여러 레퍼런스의 latent를 앙상블(평균)하거나, best reference 선택.
- **프롬프트 준수(CLIP-T) 개선 방법**:
  - guidance_scale(CFG) 조정, negative_prompt 최적화, scheduler 선택(Euler vs Heun), step 수.
  - 프롬프트 수정 허용(발표 명시): 예를 들어 생성 시에는 `photo of a {}` 대신 class 단어 포함 강화 등.
- **실험 절차**:
  1. 샘플 서브젝트 1~2개에서 하이퍼파라미터/기법 튜닝 후 CLIP 점수 측정.
  2. 점수 향상이 입증된 최적 설정을 전체 10개 서브젝트로 횡전개 실행.

**Phase 3 — 아이디어 (40% 비중, 차별화 핵심)**
- 실습 외 창의적 방법론 최소 1~2개 설계·실험. 예시 후보:
  - **다중 레퍼런스 통합 inversion**: 레퍼런스 전부의 latent를 결합(평균/최적 결합)해 단일 대표 latent 생성.
  - **RF-Inversion 개량**: `tau/eta` 스케줄을 step마다 adaptive하게, 또는 per-subject 최적화.
  - **프롬프트 임베딩 보간/최적화**: CLIP-T를 높이도록 프롬프트 임베딩 미세 조정.
  - **Textual Inversion + RF-Inversion 결합** 하이브리드.
  - **negative prompt 자동 설계** 등.
- **통일 방법론 vs 개별 방법론** 두 세팅의 점수를 모두 측정해 비교 (발표 요구사항).

**Phase 4 — 정성 평가(다양성) 개선**
- 동일 latent에서도 서로 다른 seed/약간의 노이즈로 다양성 확보.
- 레퍼런스 "그대로 복사" 수준이 아닌, 새로운 장면 생성 확인 (과도한 reconstruction 회피).

**Phase 5 — 최종 산출물**
- 100장 생성 (10 subjects × 10), `0.png~9.png`.
- CLIP-I/CLIP-T 서브젝트별 평균 + 전체 평균 (22개 값) 집계.
- 발표 슬라이드: 실습 영역 vs 아이디어 영역 구분, 방법론별 점수 비교, 대표 이미지 첨부.
- 이미지 zip 제출.

### 3.2 작업 분배 및 실험 효율화 (Fast Iteration)
- **샘플 1~2개 집중 검증 후 횡전개**:
  - 아이디어 시도 시 매번 100장을 생성하지 않고, **샘플 서브젝트 1~2개(10~20장)**로 빠른 피드백 루프 수행.
- **Colab A100 병렬화**:
  - 검증 완료된 방법론을 10개 서브젝트로 횡전개 시 서브젝트 단위 병렬 실행 스크립트 활용.
- `evaluation.py`는 서브젝트 1개씩 실행하므로 스크립트로 10개 루프 처리 가능.

### 3.3 리스크 / 주의사항
1. **SD3.5-medium의 diffusers LoRA/Textual Inversion 지원 여부**는 반드시 사전 검증 (SD3.5는 일부 기능 미지원 이슈 존재). 미지원이면 inversion 기반 + 데이터 증강 위주로.
2. **파일명 정렬**: 반드시 `0.png`~`9.png` zero-padding 없이 0~9 (1자리)로 맞출 것. `10.png` 금지.
3. **HuggingFace 게이팅**: 로컬은 `.env`의 `HF_TOKEN`으로 로그인 (`.env` 파일 존재 확인 완료). Colab에서는 직접 `huggingface_hub.login()` 또는 Colab Secret으로 등록.
4. **프롬프트 순서 매핑**: 생성 루프에서 i번째 프롬프트 → i.png가 되도록 인덱스 관리.
5. **CLIP-I는 전체 쌍 평균**: 레퍼런스와 생성 이미지의 전반적 유사도가 중요.
6. **아이디어(40%)**: 단순 튜닝으로는 한계, 반드시 창의적 방법론 + 그 근거/실험 결과 필요.
7. **프롬프트 폴더 경로**: `evaluation.py` 실행 시 `--prompts ./prompt` (단수형 `prompt`) 지정 확인.
8. **Colab A100 GPU 수치 안정성**: `torch.bfloat16` dtype 사용 권장 (float16 대비 언더플로우 방지).

---

## 부록 A: 전체 프롬프트 (10개 서브젝트)

### actionfigure_2 (class: action figure)
1. photo of a {}.
2. The {}, surrounded by towering skyscrapers.
3. The {} stands atop a snowy mountain peak, overlooking a vast icy landscape.
4. In a lush forest stands {}.
5. {} in times square.
6. {} on a sandy beach, with waves in the background.
7. {} riding a motorcycle.
8. {} riding a flying broom.
9. {} holding a futuristic energy sword.
10. The {} rides a majestic dragon.

### decoritems_woodenpot (wooden pot)
1. Photo of a {}.
2. {} in grand canyon.
3. {} with mountains and sunset in the background.
4. {} floating in a pool.
5. A wide shot of {} in times square.
6. {} and chocolate cake on a table.
7. Rose flowers in {} on a table.
8. Marigold flowers in the {}.
9. The {} at the entrance to a medieval castle.
10. {} with pens in it.

### furniture_sofa2 (sofa)
1. Photo of a {}.
2. {} near a pool.
3. {} at a beach with a view of the seashore.
4. {} in a garden.
5. {} in grand canyon.
6. {} in front of a medieval castle.
7. {} and a coffee table.
8. floor lamp on the side of {}.
9. {} and an orange sofa.
10. {} and a table with chocolate cake on it.

### instrument_music2 (guitar)
1. photo of a {}.
2. A {} resting against a rustic wall.
3. A {} glowing under the disco lights.
4. A {} at display in a music shop.
5. A {} resting on the sandy ocean floor with a turtle swimming by.
6. A cardinal is sitting beside the {}.
7. C-3PO playing with the {}.
8. {} beside a towering redwood tree.
9. A neon {} in a rainy, Blade Runner-style cityscape.
10. A vintage {} at a train station, a suitcase lying next to it.

### luggage_backpack1 (backpack)
1. photo of a {}.
2. {} on a rustic wooden dock overlooking a peaceful lake.
3. {} on a café table with a steaming cup of coffee nearby.
4. A {} on a trail in a pine forest.
5. A {} on a bicycle parked in a tulip field.
6. {} on the glass table of a high-rise penthouse overlooking a stunning city skyline.
7. A {} with the night sky.
8. A {} lying beside a bookshelf in a study room.
9. A person is walking with {} in hand.
10. A cat is looking from inside the {}.

### person_3 (person)
1. photo of a {}.
2. {} selfie standing under the pink blossoms of a cherry tree.
3. {} in a chef's outfit, cooking in a kitchen.
4. {} paddling a canoe on a tranquil lake.
5. {} playing with their pet dog.
6. Photo of {} taking a shot in basketball.
7. {} selfie with eiffel tower in the background.
8. {} in an astronaut suit, floating in a spaceship.
9. {} dressed in a firefighter's outfit, a raging forest fire in the background.
10. {} wearing Victorian-era clothing, reading a book in a classic British library.

### pet_cat5 (cat)
1. Photo of a {}.
2. {} swimming in a pool.
3. {} at a beach with a view of the seashore.
4. {} sitting on a window.
5. {} in times square.
6. {} is wearing sunglasses.
7. {} wearing a construction outfit.
8. {} is playing with a ball.
9. {} is wearing headphones.
10. {} oil painting ghibli inspired.

### scene_waterfall (waterfall)
1. photo of a {}.
2. The {} at dusk with the first rays of sunlight creeping in.
3. {} at night full of stars.
4. A frozen {} in the winter season and snow all around.
5. A {} in a neon-lit cyberpunk cityscape.
6. A golden retriever in front of the {}.
7. A cat sitting in front of the {}.
8. {} with a vibrant rainbow arching across its mist.
9. A flock of flamingos standing in the shallow waters of the {}.
10. A painter painting the scene of the {} on canvas.

### transport_tank (tank)
1. photo of a {}.
2. {} at the edge Grand Canyon, with the sun setting in the background.
3. A graffiti-covered {} parked on a street.
4. Photo of snow-covered {} in a forest.
5. The {} in a rustic barnyard.
6. The {} in a cherry blossom park.
7. A {} under a clear night sky.
8. A squirrel perched on the {}.
9. {} on Mars, an astronaut planting a flag nearby.
10. Photo of a drone hovering beside the {}.

### wearable_jacket1 (jacket)
1. photo of a {}.
2. {} on a rustic wooden bench in a park.
3. {} hanging on a golden coat rack.
4. {} resting on the back of a chair in a cafe.
5. A {} on a hanger in the window of a shop.
6. A {} on a hook in a study room with bookshelves.
7. {} placed on a beach chair on a sunny Caribbean beach.
8. A humanoid robot wearing {} with a futuristic city in the background.
9. {} hanging from a wall hook.
10. A mannequin showcasing {}.

---

## 부록 B: 핵심 코드 참조

### B.1 모델 로드 + 생성 (baseline)
```python
import os
from dotenv import load_dotenv
load_dotenv()  # .env에서 HF_TOKEN 로드 (로컬)

import torch
from diffusers import StableDiffusion3Pipeline
pipeline = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    text_encoder_3=None, tokenizer_3=None,
    torch_dtype=torch.bfloat16,  # Colab A100 등 Ampere GPU 권장 (float16 사용 가능)
)
pipeline.enable_model_cpu_offload()

image = pipeline(
    prompt="photo of a cat",
    negative_prompt="",
    num_inference_steps=28,
    height=512, width=512,
    guidance_scale=7.0,
    generator=torch.Generator().manual_seed(0),
).images[0]
image.save("0.png")
```

### B.2 Inversion 파이프라인 (핵심 흐름)
1. 레퍼런스 이미지 → VAE latent 인코딩 (`encode_image_to_sd3_latent`).
2. inversion scheduler (`EulerInversion` / `HeunInversion` / `ControlledODEInversion`)로 latent 역변환 (`guidance_scale=1.0`, `output_type="latent"`, null-text 또는 소스 프롬프트).
3. 생성 scheduler (`FlowMatchEulerDiscreteScheduler` / `ControlledODE`)로 복원 + 프롬프트별 생성 (`guidance_scale=7.0`, `latents=inverted`).
4. `image.save(f"{i}.png")`.

### B.3 평가 실행
```bash
python evaluation.py --dataset ./dataset --prompts ./prompt --concept <subject> --images ./generated/<subject>
```

---

## 부록 C: 원본 문서 출처
- `docs/guidance.txt` — 프로젝트 공지 (평가/제출/제약).
- `docs/PDF/ProjectOverview.pdf` — 프로젝트 개요, 10개 서브젝트, CLIP 평가, Euler Inversion 샘플.
- `docs/PDF/Day1-1.pdf` — Diffusers, SD3(Flow Matching/MMDiT), 스케줄러, CFG.
- `docs/PDF/Day1-2.pdf` — Inversion(Euler/Heun), RF-Inversion(Rout et al. ICLR 2025), Controlled ODE.
- `evaluation.py` — 평가 스크립트 (CLIP-B/32).
- `Day1_diffusers_tutorial.ipynb`, `Day1_RF_Inversion_Practice_exercise.ipynb` — 실습 코드.
