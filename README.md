# Toss NEXT ML Challenge: CTR 예측 모델링

### **Competition Period**
### [Competition Link](https://dacon.io/competitions/official/toss-next-ml-challenge)
- 광고 클릭률(CTR) 예측 AI 알고리즘 개발
- 사용자 행동 데이터를 기반으로 광고 클릭 여부를 예측하는 이진 분류 문제

- 평가 지표: **0.5 × AP (Average Precision) + 0.5 × (1 - Weighted Log Loss)**

- 주최: Toss, 데이콘
---

### **주요 특징**

- **다중 모델 앙상블**: **CatBoost**, **HistGradientBoosting**, **XGBoost** 세 가지 부스팅 모델을 앙상블하여 예측 성능을 극대화했습니다.
- **5-Fold Stratified Cross Validation with Seed Ensemble**: 각 모델별로 5개의 서로 다른 시드를 사용하여 5-Fold CV를 수행 (총 25개 모델/모델당)하여 안정적인 예측을 확보했습니다.
- **클래스 불균형 대응**: 다운샘플링 기법과 클래스 가중치 조정을 통해 불균형한 클릭 데이터셋에 효과적으로 대응했습니다.
- **Target Encoding with K-Fold**: 누수를 방지하면서 범주형 변수의 예측력을 극대화하는 K-Fold Target Encoding 기법을 적용했습니다.
- **시간대별 피처 엔지니어링**: 시간대(심야, 오전, 오후 등)를 기반으로 한 사용자 행동 패턴 분석 및 파생변수 생성했습니다.
- **Optuna 하이퍼파라미터 튜닝**: 각 모델별로 Optuna를 활용하여 최적의 하이퍼파라미터를 탐색했습니다.

---

### **개발 환경**

- **운영체제**: macOS
- **언어**: Python 3.11
- **주요 라이브러리**: CatBoost, XGBoost, HistGradientBoosting, Optuna, scikit-learn

---

### **프로젝트 구조**

```
Toss_NEXT_ML_CHALLENGE/
├── 전처리.ipynb                      # 데이터 로드 및 기본 EDA
├── inference.ipynb            # 추론 및 앙상블 코드
├── cat/
│   ├── cat_train.ipynb              # CatBoost 모델 학습
│   ├── catboost_fixed_seedEnsemble2/  # CatBoost 학습된 모델 파일들 (25개)
│   └── submit_catboost_FIXED_seedEnsemble_*.csv
├── hist/
│   ├── hist_train.ipynb             # HistGradientBoosting 모델 학습
│   ├── hgb_fixed_seedEnsemble/       # HGB 학습된 모델 파일들 (25개)
│   └── submit_HGB_FIXED_seedEnsemble_*.csv
├── xgb/
│   ├── xgb_train.ipynb              # XGBoost 모델 학습
│   ├── xgb_fixed_seedEnsemble/       # XGBoost 학습된 모델 파일들 (25개)
│   └── pred_xgb_infer_*.csv
├── README.md           
└── requirements.txt    
```

---

### **모델링 접근법**

사용자 행동 데이터를 바탕으로 광고 클릭 여부를 예측하기 위해 세 가지 부스팅 모델(CatBoost, HistGradientBoosting, XGBoost)을 활용한 앙상블 전략을 구축했습니다.

***

### 1. 데이터 전처리 파이프라인
- **다운샘플링**: clicked=0인 데이터를 clicked=1 대비 2배로 샘플링하여 클래스 불균형 완화
- **범주형 데이터 인코딩**: Label Encoding 및 K-Fold Target Encoding 적용
- **결측치 처리**: 수치형 변수는 적절한 방법으로 대체
- **시간대별 파생변수 생성**: 
  - 심야 시간대(자정~05시) 여부
  - 시간대별 사용자 행동 패턴
  - seq 기반 순서 정보 활용

### 2. 모델 학습 전략
- **세 가지 부스팅 모델 사용**: CatBoost, HistGradientBoosting(sklearn), XGBoost
- **5-Fold Stratified Cross Validation with Seed Ensemble**: 
  - 각 모델당 5개의 다른 시드(seed 0~4) 사용
  - 각 시드당 5-Fold CV 수행
  - 총 25개 모델/모델당 학습
- **Optuna 하이퍼파라미터 튜닝**: 각 모델별로 최적 파라미터 탐색
- **Early Stopping**: 과적합 방지 (CatBoost: 100 rounds)
- **Custom Evaluation Metric**: 0.5×AP + 0.5×(1-WLL) 기준으로 모델 평가

### 3. 앙상블 전략
- 각 모델의 25개 fold 예측 결과를 평균내어 단일 모델 예측값 생성
- CatBoost, HistGradientBoosting, XGBoost 세 모델의 예측값을 최종 앙상블
- 가중 평균 또는 단순 평균을 통한 최종 예측

---

### **파일 설명**

#### `전처리.ipynb`
- **데이터 로딩**: train.parquet, test.parquet 불러오기
- **기본 EDA**: 데이터 구조 파악 및 결측치 확인
- 주최측의 변수 정보 비공개로 인한 제한적 분석

#### `cat/cat_train.ipynb`
- **CatBoost 모델 학습**:
  - 5개의 다른 시드(0~4)로 각각 5-Fold CV 수행
  - Optuna를 활용한 하이퍼파라미터 튜닝 (40 trials)
  - Early Stopping: 100 rounds
  - 다운샘플링을 통한 클래스 불균형 완화
  - K-Fold Target Encoding 적용
- **모델 저장**: 총 25개 모델 파일(.cbm) 저장
- **평가 지표**: 0.5×AP + 0.5×(1-WLL)
- **제출 파일 생성**: 25개 fold 예측의 평균

#### `hist/hist_train.ipynb`
- **HistGradientBoosting 모델 학습** (sklearn):
  - 동일한 5-seed × 5-fold 전략
  - 각 시드별 최적 파라미터 적용
  - 클래스 가중치 조정
- **모델 저장**: 총 25개 모델 파일(.joblib) 저장
- **제출 파일 생성**: 25개 fold 예측의 평균

#### `xgb/xgb_train.ipynb`
- **XGBoost 모델 학습**:
  - 동일한 5-seed × 5-fold 전략
  - scale_pos_weight를 통한 클래스 불균형 대응
- **모델 저장**: 총 25개 모델 파일(.json) 저장
- **제출 파일 생성**: 25개 fold 예측의 평균

#### `inference 복사본.ipynb`
- **최종 앙상블 추론**:
  - CatBoost, HistGradientBoosting, XGBoost 세 모델의 예측값 로드
  - 앙상블 전략 적용 (가중 평균 또는 단순 평균)
  - 최종 제출 파일 생성

#### `데이터 파일`
- `train.parquet`: 학습 데이터 (사용자 행동 데이터, clicked 타겟 변수)
- `test.parquet`: 테스트 데이터 (ID 제외하고 사용)
- `sample_submission.csv`: 제출 파일 템플릿


