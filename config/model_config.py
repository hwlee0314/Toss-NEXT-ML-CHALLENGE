# 모델 하이퍼파라미터 설정

# CatBoost 설정
CATBOOST_PARAMS = {
    'iterations': 10000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 10,
    'random_seed': 42,
    'eval_metric': 'AUC',
    'verbose': False,
    'early_stopping_rounds': 100,
    'od_type': 'Iter',
    'od_wait': 100
}

# HistGradientBoosting 설정
HGB_PARAMS = {
    'max_iter': 10000,
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_samples_leaf': 20,
    'l2_regularization': 0.1,
    'random_state': 42,
    'early_stopping': True,
    'n_iter_no_change': 100,
    'validation_fraction': 0.1
}

# XGBoost 설정
XGB_PARAMS = {
    'n_estimators': 10000,
    'learning_rate': 0.05,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 100,
    'eval_metric': 'auc'
}

# 교차 검증 설정
CV_CONFIG = {
    'n_splits': 5,
    'shuffle': True,
    'random_states': [0, 1, 2, 3, 4]  # 5개의 다른 시드 사용
}

# 데이터 전처리 설정
PREPROCESSING_CONFIG = {
    'downsample_ratio': 2,  # clicked=0 : clicked=1 = 2:1
    'target_encoding_folds': 5,
    'label_encoding_columns': [],  # 실제 컬럼명으로 업데이트 필요
    'categorical_columns': [],     # 실제 컬럼명으로 업데이트 필요
    'exclude_columns': ['ID', 'clicked']
}

# 앙상블 가중치
ENSEMBLE_WEIGHTS = {
    'catboost': 1.0,
    'hgb': 1.0,
    'xgb': 1.0
}

# 파일 경로 설정
PATHS = {
    'data_dir': 'data/',
    'model_dir': 'models/',
    'output_dir': 'outputs/',
    'notebook_dir': 'notebooks/',
    'train_file': 'data/train.parquet',
    'test_file': 'data/test.parquet',
    'submission_template': 'data/sample_submission.csv'
}

# Optuna 하이퍼파라미터 튜닝 설정
OPTUNA_CONFIG = {
    'n_trials': 100,
    'timeout': 3600,  # 1시간
    'sampler': 'TPE',  # Tree-structured Parzen Estimator
    'pruner': 'MedianPruner'
}