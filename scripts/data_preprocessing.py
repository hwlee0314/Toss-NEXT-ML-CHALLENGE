"""
데이터 전처리 관련 유틸리티 함수들
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data(train_path='data/train.parquet', test_path='data/test.parquet'):
    """
    데이터를 로드하는 함수
    
    Parameters:
    -----------
    train_path : str
        훈련 데이터 파일 경로
    test_path : str
        테스트 데이터 파일 경로
    
    Returns:
    --------
    train_df, test_df : pd.DataFrame
        로드된 훈련 및 테스트 데이터
    """
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df

def downsample_data(df, target_col='clicked', ratio=2, random_state=42):
    """
    클래스 불균형 해결을 위한 다운샘플링
    
    Parameters:
    -----------
    df : pd.DataFrame
        원본 데이터
    target_col : str
        타겟 컬럼명
    ratio : int
        clicked=0 : clicked=1 비율
    random_state : int
        랜덤 시드
    
    Returns:
    --------
    pd.DataFrame
        다운샘플링된 데이터
    """
    clicked_1 = df[df[target_col] == 1]
    clicked_0 = df[df[target_col] == 0]
    
    # clicked=1의 ratio배만큼 clicked=0 샘플링
    clicked_0_sample = clicked_0.sample(n=len(clicked_1) * ratio, random_state=random_state)
    
    # 합치기
    balanced_df = pd.concat([clicked_1, clicked_0_sample], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Original shape: {df.shape}")
    print(f"Balanced shape: {balanced_df.shape}")
    print(f"Class distribution:\n{balanced_df[target_col].value_counts()}")
    
    return balanced_df

def kfold_target_encoding(df, categorical_cols, target_col, n_splits=5, random_state=42):
    """
    K-Fold Target Encoding
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    categorical_cols : list
        범주형 변수 리스트
    target_col : str
        타겟 변수명
    n_splits : int
        폴드 수
    random_state : int
        랜덤 시드
    
    Returns:
    --------
    pd.DataFrame
        인코딩된 데이터프레임
    """
    df_encoded = df.copy()
    
    # StratifiedKFold 설정
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for col in categorical_cols:
        # 새로운 컬럼명
        new_col = f'{col}_target_encoded'
        df_encoded[new_col] = 0.0
        
        # 각 폴드별로 target encoding
        for train_idx, val_idx in skf.split(df_encoded, df_encoded[target_col]):
            # 훈련 세트에서 평균 계산
            target_mean = df_encoded.iloc[train_idx].groupby(col)[target_col].mean()
            
            # 전체 평균으로 결측값 처리
            global_mean = df_encoded.iloc[train_idx][target_col].mean()
            
            # 검증 세트에 적용
            df_encoded.loc[val_idx, new_col] = df_encoded.loc[val_idx, col].map(target_mean).fillna(global_mean)
    
    return df_encoded

def apply_label_encoding(train_df, test_df, categorical_cols):
    """
    Label Encoding 적용
    
    Parameters:
    -----------
    train_df, test_df : pd.DataFrame
        훈련 및 테스트 데이터
    categorical_cols : list
        범주형 변수 리스트
    
    Returns:
    --------
    train_df, test_df : pd.DataFrame
        인코딩된 데이터프레임
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        
        # 훈련 + 테스트 데이터를 합쳐서 피팅
        combined_data = pd.concat([train_encoded[col], test_encoded[col]], ignore_index=True)
        le.fit(combined_data.astype(str))
        
        # 각각 변환
        train_encoded[col] = le.transform(train_encoded[col].astype(str))
        test_encoded[col] = le.transform(test_encoded[col].astype(str))
    
    return train_encoded, test_encoded

def create_time_features(df, time_col='time'):
    """
    시간 관련 피처 생성
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    time_col : str
        시간 컬럼명
    
    Returns:
    --------
    pd.DataFrame
        시간 피처가 추가된 데이터프레임
    """
    df_time = df.copy()
    
    # 시간을 datetime으로 변환 (필요한 경우)
    if df_time[time_col].dtype == 'object':
        df_time[time_col] = pd.to_datetime(df_time[time_col])
    
    # 시간대 피처
    df_time['hour'] = df_time[time_col].dt.hour
    df_time['day_of_week'] = df_time[time_col].dt.dayofweek
    df_time['is_weekend'] = (df_time['day_of_week'] >= 5).astype(int)
    
    # 심야 시간대 (자정~오전 5시)
    df_time['is_late_night'] = ((df_time['hour'] >= 0) & (df_time['hour'] < 5)).astype(int)
    
    # 시간대 구분
    df_time['time_period'] = pd.cut(df_time['hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['dawn', 'morning', 'afternoon', 'evening'],
                                  include_lowest=True)
    
    return df_time

def get_feature_columns(df, exclude_cols=['ID', 'clicked']):
    """
    모델링에 사용할 피처 컬럼 추출
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    exclude_cols : list
        제외할 컬럼 리스트
    
    Returns:
    --------
    list
        피처 컬럼 리스트
    """
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols