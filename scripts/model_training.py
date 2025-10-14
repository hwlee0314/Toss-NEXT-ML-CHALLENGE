"""
모델 학습 및 평가 관련 유틸리티 함수들
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, log_loss
import joblib
import json
import os
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')

def competition_metric(y_true, y_pred):
    """
    대회 평가 지표: 0.5 * AP + 0.5 * (1 - Weighted Log Loss)
    
    Parameters:
    -----------
    y_true : array-like
        실제 값
    y_pred : array-like
        예측 확률
    
    Returns:
    --------
    float
        대회 점수
    """
    ap = average_precision_score(y_true, y_pred)
    
    # Weighted Log Loss 계산
    # 클래스 가중치 계산 (clicked=1이 더 적으므로 가중치 높임)
    class_weights = len(y_true) / (2 * np.bincount(y_true))
    sample_weights = np.array([class_weights[int(y)] for y in y_true])
    
    wll = log_loss(y_true, y_pred, sample_weight=sample_weights)
    
    score = 0.5 * ap + 0.5 * (1 - wll)
    return score

def train_catboost_cv(X, y, n_splits=5, random_states=[0, 1, 2, 3, 4], 
                      model_params=None, save_dir='models/cat_model'):
    """
    CatBoost 5-Fold CV with multiple seeds
    
    Parameters:
    -----------
    X : pd.DataFrame
        피처 데이터
    y : pd.Series
        타겟 데이터
    n_splits : int
        폴드 수
    random_states : list
        사용할 시드 리스트
    model_params : dict
        모델 하이퍼파라미터
    save_dir : str
        모델 저장 디렉토리
    
    Returns:
    --------
    dict
        학습 결과 (모델들, 점수들)
    """
    if model_params is None:
        model_params = {
            'iterations': 10000,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 10,
            'random_seed': 42,
            'eval_metric': 'AUC',
            'verbose': False,
            'early_stopping_rounds': 100
        }
    
    os.makedirs(save_dir, exist_ok=True)
    
    models = {}
    scores = {}
    
    for seed in random_states:
        print(f"\n=== Seed {seed} ===")
        model_params['random_seed'] = seed
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_models = {}
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"Fold {fold}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 모델 학습
            model = CatBoostClassifier(**model_params)
            model.fit(X_train, y_train, 
                     eval_set=(X_val, y_val),
                     verbose=False)
            
            # 예측 및 평가
            y_pred = model.predict_proba(X_val)[:, 1]
            score = competition_metric(y_val, y_pred)
            
            print(f"  Score: {score:.6f}")
            
            # 모델 저장
            model_path = f"{save_dir}/catboost_fixed_seed{seed}_fold{fold}.cbm"
            model.save_model(model_path)
            
            fold_models[f'fold_{fold}'] = model
            fold_scores.append(score)
        
        models[f'seed_{seed}'] = fold_models
        scores[f'seed_{seed}'] = fold_scores
        
        print(f"Seed {seed} Average Score: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    
    return {'models': models, 'scores': scores}

def train_hgb_cv(X, y, n_splits=5, random_states=[0, 1, 2, 3, 4], 
                 model_params=None, save_dir='models/hist_model'):
    """
    HistGradientBoosting 5-Fold CV with multiple seeds
    """
    if model_params is None:
        model_params = {
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
    
    os.makedirs(save_dir, exist_ok=True)
    
    models = {}
    scores = {}
    
    for seed in random_states:
        print(f"\n=== Seed {seed} ===")
        model_params['random_state'] = seed
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_models = {}
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"Fold {fold}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 모델 학습
            model = HistGradientBoostingClassifier(**model_params)
            model.fit(X_train, y_train)
            
            # 예측 및 평가
            y_pred = model.predict_proba(X_val)[:, 1]
            score = competition_metric(y_val, y_pred)
            
            print(f"  Score: {score:.6f}")
            
            # 모델 저장
            model_path = f"{save_dir}/hgb_fixed_seed{seed}_fold{fold}.joblib"
            joblib.dump(model, model_path)
            
            fold_models[f'fold_{fold}'] = model
            fold_scores.append(score)
        
        models[f'seed_{seed}'] = fold_models
        scores[f'seed_{seed}'] = fold_scores
        
        print(f"Seed {seed} Average Score: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    
    return {'models': models, 'scores': scores}

def train_xgb_cv(X, y, n_splits=5, random_states=[0, 1, 2, 3, 4], 
                 model_params=None, save_dir='models/xgb_model'):
    """
    XGBoost 5-Fold CV with multiple seeds
    """
    if model_params is None:
        model_params = {
            'n_estimators': 10000,
            'learning_rate': 0.05,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 100
        }
    
    os.makedirs(save_dir, exist_ok=True)
    
    models = {}
    scores = {}
    
    for seed in random_states:
        print(f"\n=== Seed {seed} ===")
        model_params['random_state'] = seed
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_models = {}
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"Fold {fold}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 모델 학습
            model = xgb.XGBClassifier(**model_params)
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     verbose=False)
            
            # 예측 및 평가
            y_pred = model.predict_proba(X_val)[:, 1]
            score = competition_metric(y_val, y_pred)
            
            print(f"  Score: {score:.6f}")
            
            # 모델 저장
            model_path = f"{save_dir}/xgb_fixed_seed{seed}_fold{fold}.json"
            model.save_model(model_path)
            
            fold_models[f'fold_{fold}'] = model
            fold_scores.append(score)
        
        models[f'seed_{seed}'] = fold_models
        scores[f'seed_{seed}'] = fold_scores
        
        print(f"Seed {seed} Average Score: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    
    return {'models': models, 'scores': scores}

def load_and_predict(model_dir, X_test, model_type='catboost'):
    """
    저장된 모델들을 로드하여 예측
    
    Parameters:
    -----------
    model_dir : str
        모델이 저장된 디렉토리
    X_test : pd.DataFrame
        테스트 데이터
    model_type : str
        모델 타입 ('catboost', 'hgb', 'xgb')
    
    Returns:
    --------
    np.array
        앙상블 예측 결과
    """
    predictions = []
    
    # 모델 파일 찾기
    model_files = [f for f in os.listdir(model_dir) if f.startswith(model_type.split('_')[0])]
    model_files.sort()
    
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        
        if model_type == 'catboost':
            model = CatBoostClassifier()
            model.load_model(model_path)
        elif model_type == 'hgb':
            model = joblib.load(model_path)
        elif model_type == 'xgb':
            model = xgb.XGBClassifier()
            model.load_model(model_path)
        
        pred = model.predict_proba(X_test)[:, 1]
        predictions.append(pred)
    
    # 평균 앙상블
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred