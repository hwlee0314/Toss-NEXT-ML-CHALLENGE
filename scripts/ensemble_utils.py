"""
앙상블 및 추론 관련 유틸리티 함수들
"""
import numpy as np
import pandas as pd
import os
from glob import glob

def load_predictions(pred_dir, pattern="*.csv"):
    """
    예측 결과 파일들을 로드하여 앙상블
    
    Parameters:
    -----------
    pred_dir : str
        예측 파일들이 있는 디렉토리
    pattern : str
        파일 패턴
    
    Returns:
    --------
    pd.DataFrame
        앙상블된 예측 결과
    """
    pred_files = glob(os.path.join(pred_dir, pattern))
    pred_files.sort()
    
    predictions = []
    
    for pred_file in pred_files:
        pred_df = pd.read_csv(pred_file)
        predictions.append(pred_df['clicked'].values)
    
    # 평균 앙상블
    ensemble_pred = np.mean(predictions, axis=0)
    
    # 결과 데이터프레임 생성
    result_df = pd.DataFrame({
        'ID': pred_df['ID'],  # 마지막 파일의 ID 사용
        'clicked': ensemble_pred
    })
    
    return result_df

def weighted_ensemble(predictions, weights=None):
    """
    가중 앙상블
    
    Parameters:
    -----------
    predictions : list
        예측 결과 리스트
    weights : list
        가중치 리스트
    
    Returns:
    --------
    np.array
        가중 앙상블 결과
    """
    if weights is None:
        weights = [1.0] * len(predictions)
    
    weights = np.array(weights)
    weights = weights / weights.sum()  # 정규화
    
    weighted_pred = np.average(predictions, axis=0, weights=weights)
    return weighted_pred

def create_submission(test_ids, predictions, filename="submission.csv"):
    """
    제출 파일 생성
    
    Parameters:
    -----------
    test_ids : array-like
        테스트 데이터 ID
    predictions : array-like
        예측 확률
    filename : str
        저장할 파일명
    
    Returns:
    --------
    pd.DataFrame
        제출 데이터프레임
    """
    submission = pd.DataFrame({
        'ID': test_ids,
        'clicked': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"Submission saved to: {filename}")
    
    return submission

def ensemble_models(catboost_pred, hgb_pred, xgb_pred, weights=None):
    """
    세 모델의 예측 결과를 앙상블
    
    Parameters:
    -----------
    catboost_pred : array-like
        CatBoost 예측 결과
    hgb_pred : array-like
        HistGradientBoosting 예측 결과
    xgb_pred : array-like
        XGBoost 예측 결과
    weights : list
        각 모델의 가중치 [cat_weight, hgb_weight, xgb_weight]
    
    Returns:
    --------
    np.array
        앙상블 예측 결과
    """
    if weights is None:
        weights = [1.0, 1.0, 1.0]  # 동일 가중치
    
    predictions = [catboost_pred, hgb_pred, xgb_pred]
    ensemble_result = weighted_ensemble(predictions, weights)
    
    return ensemble_result

def evaluate_ensemble_combinations(predictions_dict, y_true=None):
    """
    다양한 앙상블 조합의 성능 평가
    
    Parameters:
    -----------
    predictions_dict : dict
        {'model_name': predictions} 형태의 딕셔너리
    y_true : array-like
        실제 값 (있는 경우)
    
    Returns:
    --------
    dict
        각 조합별 성능 결과
    """
    from itertools import combinations
    from sklearn.metrics import average_precision_score, log_loss
    
    model_names = list(predictions_dict.keys())
    results = {}
    
    # 단일 모델 성능
    if y_true is not None:
        for name in model_names:
            pred = predictions_dict[name]
            ap = average_precision_score(y_true, pred)
            logloss = log_loss(y_true, pred)
            score = 0.5 * ap + 0.5 * (1 - logloss)
            results[f'single_{name}'] = {
                'AP': ap,
                'LogLoss': logloss,
                'Competition_Score': score
            }
    
    # 두 모델 조합
    for combo in combinations(model_names, 2):
        preds = [predictions_dict[name] for name in combo]
        ensemble_pred = np.mean(preds, axis=0)
        
        if y_true is not None:
            ap = average_precision_score(y_true, ensemble_pred)
            logloss = log_loss(y_true, ensemble_pred)
            score = 0.5 * ap + 0.5 * (1 - logloss)
            results[f'ensemble_{"_".join(combo)}'] = {
                'AP': ap,
                'LogLoss': logloss,
                'Competition_Score': score
            }
    
    # 전체 모델 앙상블
    all_preds = [predictions_dict[name] for name in model_names]
    ensemble_pred = np.mean(all_preds, axis=0)
    
    if y_true is not None:
        ap = average_precision_score(y_true, ensemble_pred)
        logloss = log_loss(y_true, ensemble_pred)
        score = 0.5 * ap + 0.5 * (1 - logloss)
        results['ensemble_all'] = {
            'AP': ap,
            'LogLoss': logloss,
            'Competition_Score': score
        }
    
    return results, ensemble_pred