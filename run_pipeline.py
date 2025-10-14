#!/usr/bin/env python3
"""
전체 파이프라인 실행 스크립트
데이터 전처리부터 모델 학습, 앙상블 추론까지 전체 과정을 자동화
"""

import os
import sys
import argparse
import logging
import numpy as np
from datetime import datetime

# 프로젝트 루트를 Python path에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from scripts.data_preprocessing import (
    load_data, downsample_data, kfold_target_encoding, 
    apply_label_encoding, create_time_features, get_feature_columns
)
from scripts.model_training import train_catboost_cv, train_hgb_cv, train_xgb_cv
from scripts.ensemble_utils import load_and_predict, ensemble_models, create_submission
from config.model_config import (
    CATBOOST_PARAMS, HGB_PARAMS, XGB_PARAMS, CV_CONFIG, 
    PREPROCESSING_CONFIG, ENSEMBLE_WEIGHTS, PATHS
)

def setup_logging():
    """로깅 설정"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"outputs/pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Toss ML Challenge Pipeline')
    parser.add_argument('--stage', choices=['preprocess', 'train', 'ensemble', 'all'], 
                       default='all', help='실행할 단계')
    parser.add_argument('--models', nargs='+', choices=['catboost', 'hgb', 'xgb'], 
                       default=['catboost', 'hgb', 'xgb'], help='학습할 모델들')
    parser.add_argument('--quick-run', action='store_true', 
                       help='빠른 실행 (적은 반복, 작은 데이터셋)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs('outputs', exist_ok=True)
    logger = setup_logging()
    
    logger.info("=== Toss ML Challenge Pipeline 시작 ===")
    logger.info(f"실행 단계: {args.stage}")
    logger.info(f"선택된 모델: {args.models}")
    
    try:
        if args.stage in ['preprocess', 'all']:
            logger.info("1. 데이터 전처리 단계 시작")
            
            # 데이터 로드
            train_df, test_df = load_data(PATHS['train_file'], PATHS['test_file'])
            
            # 다운샘플링 (빠른 실행시 더 적은 비율)
            ratio = 1 if args.quick_run else PREPROCESSING_CONFIG['downsample_ratio']
            train_balanced = downsample_data(train_df, ratio=ratio)
            
            # 범주형 변수 처리 (실제 컬럼에 맞게 수정 필요)
            categorical_cols = [col for col in train_balanced.columns 
                              if col not in PREPROCESSING_CONFIG['exclude_columns']]
            
            # Label Encoding
            train_encoded, test_encoded = apply_label_encoding(
                train_balanced, test_df, categorical_cols
            )
            
            # 시간 피처 생성 (시간 컬럼이 있는 경우)
            # train_encoded = create_time_features(train_encoded)
            # test_encoded = create_time_features(test_encoded)
            
            # 피처 컬럼 추출
            feature_cols = get_feature_columns(train_encoded, PREPROCESSING_CONFIG['exclude_columns'])
            
            X_train = train_encoded[feature_cols]
            y_train = train_encoded['clicked']
            X_test = test_encoded[feature_cols]
            
            logger.info(f"전처리 완료 - 훈련: {X_train.shape}, 테스트: {X_test.shape}")
        
        if args.stage in ['train', 'all']:
            logger.info("2. 모델 학습 단계 시작")
            
            # Quick run 설정
            if args.quick_run:
                cv_config = {'n_splits': 2, 'random_states': [0, 1]}
                for params in [CATBOOST_PARAMS, HGB_PARAMS, XGB_PARAMS]:
                    params['iterations'] = 100
                    params['n_estimators'] = 100
                    params['max_iter'] = 100
            else:
                cv_config = CV_CONFIG
            
            # CatBoost 학습
            if 'catboost' in args.models:
                logger.info("CatBoost 학습 시작")
                cat_results = train_catboost_cv(
                    X_train, y_train, 
                    model_params=CATBOOST_PARAMS,
                    **cv_config
                )
                logger.info("CatBoost 학습 완료")
            
            # HistGradientBoosting 학습  
            if 'hgb' in args.models:
                logger.info("HistGradientBoosting 학습 시작")
                hgb_results = train_hgb_cv(
                    X_train, y_train,
                    model_params=HGB_PARAMS,
                    **cv_config
                )
                logger.info("HistGradientBoosting 학습 완료")
            
            # XGBoost 학습
            if 'xgb' in args.models:
                logger.info("XGBoost 학습 시작")  
                xgb_results = train_xgb_cv(
                    X_train, y_train,
                    model_params=XGB_PARAMS,
                    **cv_config
                )
                logger.info("XGBoost 학습 완료")
        
        if args.stage in ['ensemble', 'all']:
            logger.info("3. 앙상블 추론 단계 시작")
            
            predictions = {}
            
            # 각 모델 예측
            if 'catboost' in args.models:
                cat_pred = load_and_predict('models/cat_model', X_test, 'catboost')
                predictions['catboost'] = cat_pred
            
            if 'hgb' in args.models:
                hgb_pred = load_and_predict('models/hist_model', X_test, 'hgb') 
                predictions['hgb'] = hgb_pred
            
            if 'xgb' in args.models:
                xgb_pred = load_and_predict('models/xgb_model', X_test, 'xgb')
                predictions['xgb'] = xgb_pred
            
            # 앙상블
            if len(predictions) == 3:
                final_pred = ensemble_models(
                    predictions['catboost'], 
                    predictions['hgb'], 
                    predictions['xgb'],
                    weights=[ENSEMBLE_WEIGHTS['catboost'], 
                            ENSEMBLE_WEIGHTS['hgb'], 
                            ENSEMBLE_WEIGHTS['xgb']]
                )
            else:
                # 사용 가능한 모델들의 평균
                final_pred = np.mean(list(predictions.values()), axis=0)
            
            # 제출 파일 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            submission_file = f"outputs/submission_{timestamp}.csv"
            
            create_submission(test_df['ID'], final_pred, submission_file)
            logger.info(f"제출 파일 생성 완료: {submission_file}")
        
        logger.info("=== 파이프라인 완료 ===")
    
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()