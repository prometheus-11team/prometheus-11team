import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 디렉토리 생성
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =============================================================================
# 1. Transformer 모델 정의 (두 번째 코드에서 가져옴)
# =============================================================================

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return torch.sigmoid(self.head(x[:, -1]))  # (B,1)

# =============================================================================
# 2. 확장된 데이터 전처리 클래스
# =============================================================================

class EnhancedDataProcessor:
    """확장된 데이터 전처리 클래스"""
    
    def __init__(self, transformer_model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transformer_model = None
        self.scaler = StandardScaler()
        
        # Transformer 모델 로드
        if transformer_model_path and os.path.exists(transformer_model_path):
            self.load_transformer_model(transformer_model_path)
    
    def load_transformer_model(self, model_path):
        """Transformer 모델 로드"""
        try:
            self.transformer_model = TransformerClassifier(input_dim=17).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            self.transformer_model.load_state_dict(state_dict)
            self.transformer_model.eval()
            print("✅ Transformer 모델 로드 완료")
        except Exception as e:
            print(f"❌ Transformer 모델 로드 실패: {e}")
            self.transformer_model = None
    
    def load_and_merge_data(self, stock_data_path, macro_data_path, financial_data_path, sentiment_data_paths):
        """모든 데이터 로드 및 병합"""
        
        # 1. 주식 데이터 로드
        df_stock = pd.read_csv(stock_data_path)
        df_stock["date"] = pd.to_datetime(df_stock["date"])
        df_stock = df_stock.rename(columns={"ticker": "tic"})
        df_stock = df_stock.sort_values(["date", "tic"]).reset_index(drop=True)
        
        # 2. 거시경제 지표 로드
        if os.path.exists(macro_data_path):
            df_macro = pd.read_csv(macro_data_path)
            df_macro["date"] = pd.to_datetime(df_macro["date"])
            df_macro = df_macro.sort_values("date").reset_index(drop=True)
        else:
            print("거시경제 데이터가 없어 기본값으로 설정")
            df_macro = pd.DataFrame({
                'date': df_stock['date'].unique(),
                'federal_funds_rate': 0.0,
                'treasury_yield': 0.0,
                'cpi': 0.0,
                'unemployment_rate': 0.0
            })
        
        # 3. 재무 데이터 로드 및 전처리
        if os.path.exists(financial_data_path):
            df_financial = self._process_financial_data(financial_data_path, df_stock)
        else:
            print("재무 데이터가 없어 기본값으로 설정")
            df_financial = df_stock[['date', 'tic']].copy()
            for col in ['Operating Income', 'Net Income', 'EPS Diluted', 'Total Assets', 'Shareholders Equity']:
                df_financial[col] = 0.0
        
        # 4. 감정 분석 데이터 로드
        df_sentiment = self._load_sentiment_data(sentiment_data_paths, df_stock)
        
        # 5. 모든 데이터 병합
        df_merged = self._merge_all_data(df_stock, df_macro, df_financial, df_sentiment)
        
        # 6. Transformer 예측 추가
        df_merged = self._add_transformer_predictions(df_merged)
        
        return df_merged
    
    def _process_financial_data(self, financial_path, df_stock):
        """재무 데이터 전처리"""
        try:
            financial_data = pd.read_csv(financial_path)
            
            # 회사명-티커 매핑
            company_to_ticker = {
                'Alphabet': 'GOOGL', 'Amazon': 'AMZN', 'Apple': 'AAPL',
                'Meta': 'META', 'Microsoft': 'MSFT', 'Nvidia': 'NVDA', 'Tesla': 'TSLA'
            }
            
            financial_data['tic'] = financial_data['Company'].map(company_to_ticker)
            financial_data['date'] = pd.to_datetime(financial_data['Release Date'])
            
            features = ['Operating Income', 'Net Income', 'EPS Diluted', 'Total Assets', 'Shareholders Equity']
            daily_index = pd.date_range(start=df_stock['date'].min(), end=df_stock['date'].max(), freq='D')
            
            daily_financial_list = []
            for ticker in financial_data['tic'].unique():
                if pd.isna(ticker):
                    continue
                df_t = financial_data.loc[financial_data['tic'] == ticker, ['date', 'tic'] + features].copy()
                df_t = df_t.set_index('date').sort_index()
                df_t = df_t[~df_t.index.duplicated(keep='last')]
                df_t = df_t.reindex(daily_index).ffill()
                df_t['tic'] = ticker
                df_t = df_t.reset_index().rename(columns={'index': 'date'})
                daily_financial_list.append(df_t)
            
            return pd.concat(daily_financial_list, ignore_index=True)
            
        except Exception as e:
            print(f"재무 데이터 처리 실패: {e}")
            df_financial = df_stock[['date', 'tic']].copy()
            for col in ['Operating Income', 'Net Income', 'EPS Diluted', 'Total Assets', 'Shareholders Equity']:
                df_financial[col] = 0.0
            return df_financial
    
    def _load_sentiment_data(self, sentiment_data_paths, df_stock):
        """감정 분석 데이터 로드"""
        sentiment_dfs = []
        
        for path in sentiment_data_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    
                    # 날짜 컬럼 정리
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    elif 'datadate' in df.columns:
                        df['date'] = pd.to_datetime(df['datadate'])
                    
                    # 감정 점수 정리
                    if 'sentiment_score' in df.columns:
                        df['positive'] = df['sentiment_score'].apply(lambda x: max(0, x))
                        df['negative'] = df['sentiment_score'].apply(lambda x: max(0, -x))
                        df['neutral'] = df['sentiment_score'].apply(lambda x: 1 - abs(x) if abs(x) <= 1 else 0)
                    elif not all(col in df.columns for col in ['positive', 'negative', 'neutral']):
                        df['positive'] = 0.5
                        df['negative'] = 0.3
                        df['neutral'] = 0.2
                    
                    # 소스 구분
                    if 'googlenews' in path:
                        df['source'] = 'google_news'
                    elif 'reddit' in path:
                        df['source'] = 'reddit'
                    else:
                        df['source'] = 'unknown'
                    
                    sentiment_dfs.append(df)
                    print(f"감정 데이터 로드: {path} - {len(df)} 행")
                    
                except Exception as e:
                    print(f"감정 데이터 로드 실패 {path}: {e}")
        
        if sentiment_dfs:
            combined_sentiment = pd.concat(sentiment_dfs, ignore_index=True)
            
            # 날짜별, 종목별로 감정 점수 집계
            sentiment_agg = combined_sentiment.groupby(['date', 'tic', 'source']).agg({
                'positive': 'mean',
                'negative': 'mean',
                'neutral': 'mean'
            }).reset_index()
            
            # 피벗하여 소스별 감정 점수 분리
            sentiment_pivot = sentiment_agg.pivot_table(
                index=['date', 'tic'], 
                columns='source', 
                values=['positive', 'negative', 'neutral'],
                fill_value=0.3
            )
            
            # 컬럼명 평탄화
            sentiment_pivot.columns = ['_'.join(col).strip() for col in sentiment_pivot.columns]
            sentiment_pivot = sentiment_pivot.reset_index()
            
            return sentiment_pivot
        else:
            # 기본 감정 데이터 생성
            unique_combinations = df_stock[['date', 'tic']].drop_duplicates()
            sentiment_cols = [
                'positive_google_news', 'negative_google_news', 'neutral_google_news',
                'positive_reddit', 'negative_reddit', 'neutral_reddit'
            ]
            for col in sentiment_cols:
                unique_combinations[col] = 0.3
            
            return unique_combinations
    
    def _merge_all_data(self, df_stock, df_macro, df_financial, df_sentiment):
        """모든 데이터 병합"""
        # 거시경제 지표 병합
        df_merged = pd.merge_asof(
            df_stock.sort_values('date'), 
            df_macro.sort_values('date'),
            on='date',
            direction='backward'
        )
        
        # 재무 데이터 병합
        df_merged = pd.merge(df_merged, df_financial, on=['date', 'tic'], how='left')
        
        # 감정 데이터 병합
        df_merged = pd.merge(df_merged, df_sentiment, on=['date', 'tic'], how='left')
        
        # 결측치 처리
        macro_cols = [col for col in df_macro.columns if col != 'date']
        if macro_cols:
            df_merged[macro_cols] = df_merged[macro_cols].fillna(method='ffill').fillna(method='bfill')
        
        financial_cols = ['Operating Income', 'Net Income', 'EPS Diluted', 'Total Assets', 'Shareholders Equity']
        for col in financial_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(0)
        
        sentiment_cols = [col for col in df_merged.columns if any(x in col for x in ['positive', 'negative', 'neutral'])]
        for col in sentiment_cols:
            df_merged[col] = df_merged[col].fillna(0.3)
        
        return df_merged
    
    def _add_transformer_predictions(self, df_merged):
        """Transformer 예측 추가"""
        if self.transformer_model is None:
            # Transformer가 없으면 기본값 추가
            df_merged['transformer_prediction'] = 0.5
            df_merged['transformer_confidence'] = 0.0
            df_merged['transformer_signal'] = 0
            return df_merged
        
        # 기술지표 컬럼 정의
        tech_indicators = ['open', 'high', 'low', 'close', 'volume']
        
        # 기존 기술지표 찾기
        available_indicators = [col for col in df_merged.columns 
                              if any(indicator in col.lower() for indicator in 
                                   ['rsi', 'macd', 'ema', 'sma', 'bb', 'roc', 'atr', 'adx', 'cci', 'willr', 'momentum', 'stoch'])]
        
        feature_cols = tech_indicators + available_indicators
        feature_cols = [col for col in feature_cols if col in df_merged.columns]
        
        if len(feature_cols) < 17:
            # 부족한 피처는 0으로 패딩
            for i in range(17 - len(feature_cols)):
                df_merged[f'padding_{i}'] = 0.0
                feature_cols.append(f'padding_{i}')
        
        transformer_predictions = []
        transformer_confidences = []
        transformer_signals = []
        
        for idx, row in df_merged.iterrows():
            try:
                # 17개 피처 추출
                features = row[feature_cols[:17]].values.astype(float)
                features = np.nan_to_num(features, nan=0.0)
                
                # 정규화
                features = np.clip(features, -100, 100)
                
                # Transformer 예측
                input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    prediction = self.transformer_model(input_tensor).cpu().numpy().flatten()[0]
                    confidence = abs(prediction - 0.5) * 2
                    signal = 1 if prediction > 0.5 else 0
                
                transformer_predictions.append(float(prediction))
                transformer_confidences.append(float(confidence))
                transformer_signals.append(signal)
                
            except Exception as e:
                transformer_predictions.append(0.5)
                transformer_confidences.append(0.0)
                transformer_signals.append(0)
        
        df_merged['transformer_prediction'] = transformer_predictions
        df_merged['transformer_confidence'] = transformer_confidences
        df_merged['transformer_signal'] = transformer_signals
        
        print(f"✅ Transformer 예측 {len(transformer_predictions)}개 추가 완료")
        df_merged = self._enhance_signals(df_merged)
        return df_merged
    
    def _enhance_signals(self, df):
        """입력 신호들을 증폭하여 더 극단적으로 만들기"""
        
        # 1. Transformer 예측 신호 증폭
        if 'transformer_prediction' in df.columns:
            df['transformer_prediction'] = df['transformer_prediction'].apply(
                lambda x: 0.5 + (x - 0.5) * 2.5 if not pd.isna(x) else 0.5
            )
            df['transformer_prediction'] = df['transformer_prediction'].clip(0.1, 0.9)
        
        # 2. 감정 분석 신호 증폭
        sentiment_cols = [col for col in df.columns if 'positive' in col or 'negative' in col]
        for col in sentiment_cols:
            if col in df.columns:
                # 중립값(0.3)에서 벗어난 정도를 2배로 증폭
                df[col] = df[col].apply(
                    lambda x: 0.3 + (x - 0.3) * 2.0 if not pd.isna(x) else 0.3
                )
                df[col] = df[col].clip(0.1, 0.8)
        
        # 3. 기술지표 정규화 (RSI, MACD 등)
        tech_cols = ['RSI_14', 'MACD', 'ROC_10']
        for col in tech_cols:
            if col in df.columns:
                # 표준화 후 극단값 강조
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
                    df[col] = df[col] * 1.5  # 신호 증폭
                    df[col] = df[col].clip(-3, 3)  # 극단값 제한
        
        print("✅ 입력 신호 증폭 완료")
        return df

# =============================================================================
# 3. 메인 실행 코드 (첫 번째 코드 구조 유지)
# =============================================================================

def main():
    print("📊 확장된 FinRL 시스템 시작...")
    
    # =============================================================================
    # 데이터 경로 설정
    # =============================================================================
    
    # 기본 데이터 경로
    stock_data_path = "/Users/gamjawon/prometheus-11team/FinRL-Library/examples/data/M7_stock_data_with_indicators.csv"
    macro_data_path = "/Users/gamjawon/prometheus-11team/DATA/technical/macro_indicators_2020_2025-03.csv"
    financial_data_path = "/Users/gamjawon/prometheus-11team/FinRL-Library/examples/data/M7_financial_data_2020_2025.csv"
    
    # 감정 분석 데이터 경로
    sentiment_data_paths = [
        "/Users/gamjawon/prometheus-11team/FinRL-Library/examples/data/M7_googlenews_2020_2022_sentiment_feature.csv",
        "/Users/gamjawon/prometheus-11team/FinRL-Library/examples/data/M7_reddits_2022_2025_sentiment_feature.csv"
    ]
    
    # Transformer 모델 경로
    transformer_model_path = "/Users/gamjawon/prometheus-11team/model/transformer_classifier_best.pt"
    
    # =============================================================================
    # 확장된 데이터 전처리
    # =============================================================================
    
    print("🔄 확장된 데이터 전처리 시작...")
    
    # 데이터 프로세서 초기화
    data_processor = EnhancedDataProcessor(transformer_model_path)
    
    # 모든 데이터 로드 및 병합
    df = data_processor.load_and_merge_data(
        stock_data_path, macro_data_path, financial_data_path, sentiment_data_paths
    )
    
    print(f"✅ 확장된 데이터 준비 완료! 총 컬럼 수: {len(df.columns)}")
    print(f"데이터 기간: {df['date'].min()} ~ {df['date'].max()}")
    print(f"새로 추가된 피처들:")
    
    # 새로 추가된 피처들 확인
    new_features = [col for col in df.columns if any(x in col for x in 
                   ['transformer', 'positive', 'negative', 'neutral', 'Operating', 'Net Income', 'EPS', 'Assets', 'Equity'])]
    for feature in new_features:
        print(f"  - {feature}")
    
    # =============================================================================
    # 기술지표 및 새 지표 정의 (첫 번째 코드 방식 유지)
    # =============================================================================
    
    # 기본 주가 데이터 컬럼
    basic_cols = ["date", "tic", "open", "high", "low", "close", "volume"]
    
    # 기존 기술지표 (정확한 컬럼명으로 수정)
    tech_indicators = ['SMA_50', 'SMA_200', 'RSI_14', 'ROC_10', 'MACD', 'MACD_Signal']
    # 실제 데이터에 존재하는지 확인 후 필터링
    tech_indicators = [col for col in tech_indicators if col in df.columns]
    
    # 거시경제 지표 (정확한 컬럼명으로 수정)
    macro_indicators = [
        'Federal Funds Rate', 
        '10Y Treasury Yield', 
        'CPI', 
        'Core CPI', 
        'PCE Price Index', 
        'Retail Sales', 
        'Unemployment Rate', 
        'Non-Farm Payrolls', 
        'M2 Money Stock'
    ]
    # 실제 데이터에 존재하는지 확인 후 필터링
    macro_indicators = [col for col in macro_indicators if col in df.columns]
    
    # Transformer 지표
    transformer_indicators = ['transformer_prediction', 'transformer_confidence', 'transformer_signal']
    
    # 감정 분석 지표
    sentiment_indicators = [col for col in df.columns if any(x in col for x in ['positive', 'negative', 'neutral'])]
    
    # 모든 지표 통합 (재무지표 제외)
    ALL_INDICATORS = tech_indicators + macro_indicators + transformer_indicators + sentiment_indicators
    
    print(f"\n📈 사용할 지표들:")
    print(f"기술지표 ({len(tech_indicators)}개): {tech_indicators}")
    print(f"거시지표 ({len(macro_indicators)}개): {macro_indicators}")
    print(f"Transformer 지표 ({len(transformer_indicators)}개): {transformer_indicators}")
    print(f"감정지표 ({len(sentiment_indicators)}개): {sentiment_indicators}")
    print(f"총 지표 수: {len(ALL_INDICATORS)}개")

    # =============================================================================
    # 데이터 분할 (첫 번째 코드와 동일)
    # =============================================================================
    
    train_df = data_split(df, start="2020-01-01", end="2023-12-31")
    test_df = data_split(df, start="2024-01-01", end="2025-04-30")
    
    print(f"\n📊 데이터 분할:")
    print(f"훈련 데이터: {train_df['date'].min()} ~ {train_df['date'].max()} ({len(train_df)} rows)")
    print(f"테스트 데이터: {test_df['date'].min()} ~ {test_df['date'].max()} ({len(test_df)} rows)")
    
    # =============================================================================
    # 데이터 전처리 (첫 번째 코드와 동일)
    # =============================================================================
    
    # 결측치 처리
    train_df = train_df.fillna(method="ffill").fillna(method="bfill")
    test_df = test_df.fillna(method="ffill").fillna(method="bfill")
    
    # 결측치 제거
    train_df = train_df.dropna(subset=ALL_INDICATORS)
    test_df = test_df.dropna(subset=ALL_INDICATORS)
    
    # 인덱스 재설정
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    train_df.index = train_df["date"].factorize()[0]
    test_df.index = test_df["date"].factorize()[0]
    
    print(f"\n✅ 전처리 완료:")
    print(f"훈련 데이터: {len(train_df)} rows")
    print(f"테스트 데이터: {len(test_df)} rows")
    
    # =============================================================================
    # 환경 설정 (확장된 상태 공간)
    # =============================================================================
    
    stock_dim = len(train_df["tic"].unique())
    state_space = 1 + 2 * stock_dim + len(ALL_INDICATORS) * stock_dim  # 확장된 상태 공간
    
    env_kwargs = {
        "hmax": 800,
        "initial_amount": 1_000_000,
        "buy_cost_pct": [0.0005] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": ALL_INDICATORS,  # 확장된 지표 리스트
        "action_space": stock_dim,
        "reward_scaling": 2e-1,
        "num_stock_shares": [0] * stock_dim,
        "turbulence_threshold": None,
        "day": 0
    }
    
    print(f"\n🎯 확장된 환경 설정:")
    print(f"주식 종목 수: {stock_dim}")
    print(f"확장된 상태 공간 크기: {state_space}")
    print(f"전체 지표 수: {len(ALL_INDICATORS)}")
    print(f"상태 공간 구성:")
    print(f"  - 기본 정보: 1 (현금)")
    print(f"  - 주식 보유량: {stock_dim}")
    print(f"  - 주식 가격: {stock_dim}")
    print(f"  - 기술/거시/재무/감정/Transformer 지표: {len(ALL_INDICATORS)} × {stock_dim}")
    
    # =============================================================================
    # 모델 학습 (첫 번째 코드와 동일하지만 확장된 입력)
    # =============================================================================
    
    print(f"\n🚀 확장된 모델 학습 시작...")
    
    env_train = StockTradingEnv(df=train_df, **env_kwargs)
    agent = DRLAgent(env=env_train)
    
    # TD3 모델 생성 및 학습
    model = agent.get_model("td3")
    trained_model = agent.train_model(
        model=model,
        tb_log_name="td3_with_enhanced_features",
        total_timesteps=150000
    )
    
    
    print("✅ 확장된 모델 학습 완료!")
    
    # 모델 저장
    model_save_path = os.path.join("models", "td3_enhanced_model(4)")
    trained_model.save(model_save_path)
    print(f"\n💾 확장된 모델이 '{model_save_path}'에 저장되었습니다!")
    
    # =============================================================================
    # 백테스팅 (첫 번째 코드와 동일)
    # =============================================================================
    
    print(f"\n📊 확장된 모델 백테스팅 시작...")
    
    env_test = StockTradingEnv(df=test_df, **env_kwargs)
    df_account_value, df_actions = agent.DRL_prediction(model=trained_model, environment=env_test)
    
    print("✅ 백테스팅 완료!")
    
    # =============================================================================
    # 성능 분석 (첫 번째 코드와 동일)
    # =============================================================================
    
    # 성능 지표 계산 함수들 (첫 번째 코드에서 그대로 가져옴)
    def calculate_total_return(df):
        start_value = df["account_value"].iloc[0]
        end_value = df["account_value"].iloc[-1]
        total_return = (end_value - start_value) / start_value
        return total_return
    
    def calculate_cagr(df):
        start_value = df["account_value"].iloc[0]
        end_value = df["account_value"].iloc[-1]
        n_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
        n_years = n_days / 365.25
        if n_years > 0:
            cagr = (end_value / start_value) ** (1 / n_years) - 1
        else:
            cagr = 0
        return cagr
    
    def calculate_mdd(df):
        cumulative = df["account_value"].cummax()
        drawdown = df["account_value"] / cumulative - 1
        mdd = drawdown.min()
        return mdd
    
    def calculate_sharpe_ratio(df, risk_free_rate=0.02):
        returns = df["account_value"].pct_change().dropna()
        excess_returns = returns - risk_free_rate/252
        if excess_returns.std() != 0:
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        return sharpe
    
    def calculate_volatility(df):
        returns = df["account_value"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        return volatility
    
    def calculate_benchmark_return(df_test, benchmark_tickers=["AAPL", "GOOGL", "MSFT"]):
        benchmark_returns = []
        
        for ticker in benchmark_tickers:
            df_ticker = df_test[df_test["tic"] == ticker].copy()
            if len(df_ticker) > 0:
                df_ticker = df_ticker.sort_values("date")
                buy_price = df_ticker.iloc[0]["close"]
                sell_price = df_ticker.iloc[-1]["close"]
                return_rate = (sell_price - buy_price) / buy_price
                benchmark_returns.append(return_rate)
        
        if benchmark_returns:
            avg_benchmark_return = np.mean(benchmark_returns)
        else:
            avg_benchmark_return = 0
            
        return avg_benchmark_return, benchmark_returns
    
    # 성과 분석
    print(f"\n" + "="*60)
    print(f"📊 확장된 모델 성과 분석 결과")
    print(f"="*60)
    
    # DRL 모델 성과
    total_return = calculate_total_return(df_account_value)
    cagr = calculate_cagr(df_account_value)
    mdd = calculate_mdd(df_account_value)
    sharpe = calculate_sharpe_ratio(df_account_value)
    volatility = calculate_volatility(df_account_value)
    
    # 벤치마크 성과
    benchmark_return, individual_returns = calculate_benchmark_return(test_df)
    
    # 결과 출력
    print(f"🤖 확장된 TD3 모델 성과:")
    print(f"   💰 총 수익률: {total_return:.2%}")
    print(f"   📈 연복리 수익률 (CAGR): {cagr:.2%}")
    print(f"   📉 최대 낙폭 (MDD): {mdd:.2%}")
    print(f"   ⚡ 샤프 비율: {sharpe:.2f}")
    print(f"   📊 변동성: {volatility:.2%}")
    
    print(f"\n📊 벤치마크 성과:")
    print(f"   🏆 평균 Buy & Hold 수익률: {benchmark_return:.2%}")
    for i, ticker in enumerate(["AAPL", "GOOGL", "MSFT"]):
        if i < len(individual_returns):
            print(f"   📈 {ticker} Buy & Hold: {individual_returns[i]:.2%}")
    
    print(f"\n🎯 상대 성과:")
    outperformance = total_return - benchmark_return
    print(f"   🚀 초과 수익률: {outperformance:.2%}")
    print(f"   📈 성과: {'승리! 🎉' if outperformance > 0 else '아쉽... 📉'}")
    
    # 최종 포트폴리오 가치
    final_value = df_account_value["account_value"].iloc[-1]
    initial_value = df_account_value["account_value"].iloc[0]
    profit = final_value - initial_value
    
    print(f"\n💎 최종 결과:")
    print(f"   🏦 초기 자금: {initial_value:,.0f}원")
    print(f"   💰 최종 자금: {final_value:,.0f}원")
    print(f"   💵 절대 수익: {profit:,.0f}원")
    
    # =============================================================================
    # 확장된 특성 분석
    # =============================================================================
    
    print(f"\n" + "="*60)
    print(f"🔍 확장된 특성 기여도 분석")
    print(f"="*60)
    
    # 각 특성 그룹별 통계
    print(f"📊 사용된 특성 그룹:")
    print(f"   🔧 기술지표: {len(tech_indicators)}개")
    print(f"   🌍 거시지표: {len(macro_indicators)}개") 
    print(f"   🤖 Transformer: {len(transformer_indicators)}개")
    print(f"   😊 감정지표: {len(sentiment_indicators)}개")
    
    # Transformer 예측 정확도 분석 (테스트 데이터에서)
    if 'transformer_prediction' in test_df.columns:
        transformer_preds = test_df['transformer_prediction'].values
        print(f"\n🤖 Transformer 예측 분석:")
        print(f"   평균 예측값: {np.mean(transformer_preds):.3f}")
        print(f"   예측 분산: {np.var(transformer_preds):.3f}")
        print(f"   강세 예측 비율: {np.mean(transformer_preds > 0.5):.1%}")
    
    # 감정 지표 분석
    sentiment_cols = [col for col in test_df.columns if 'positive' in col or 'negative' in col]
    if sentiment_cols:
        print(f"\n😊 감정 분석:")
        for col in sentiment_cols[:4]:  # 처음 4개만 출력
            if col in test_df.columns:
                avg_sentiment = test_df[col].mean()
                print(f"   {col}: {avg_sentiment:.3f}")
    
    # =============================================================================
    # 시각화 (확장된 버전)
    # =============================================================================
    
    plt.figure(figsize=(20, 12))
    
    # 1. 포트폴리오 가치 변화
    plt.subplot(3, 3, 1)
    plt.plot(df_account_value["date"], df_account_value["account_value"], 'b-', linewidth=2, label='Enhanced DRL Portfolio')
    plt.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.7, label='Initial Value')
    plt.title('📈 Enhanced Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (원)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 2. 수익률 변화
    plt.subplot(3, 3, 2)
    returns = df_account_value["account_value"].pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1
    plt.plot(df_account_value["date"][1:], cumulative_returns * 100, 'g-', linewidth=2)
    plt.title('📊 Cumulative Returns (%)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 3. 드로우다운
    plt.subplot(3, 3, 3)
    cumulative = df_account_value["account_value"].cummax()
    drawdown = (df_account_value["account_value"] / cumulative - 1) * 100
    plt.fill_between(df_account_value["date"], drawdown, 0, color='red', alpha=0.3)
    plt.plot(df_account_value["date"], drawdown, 'r-', linewidth=1)
    plt.title('📉 Drawdown (%)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 4. 거래 활동
    plt.subplot(3, 3, 4)
    if len(df_actions) > 0:
        action_cols = [col for col in df_actions.columns if col != 'date']
        daily_trades = df_actions[action_cols].abs().sum(axis=1)
        plt.plot(range(len(daily_trades)), daily_trades, 'purple', linewidth=1)
        plt.title('🔄 Daily Trading Activity')
        plt.xlabel('Time steps')
        plt.ylabel('Total Trades')
        plt.grid(True, alpha=0.3)
    
    # 5. Transformer 예측 분포
    plt.subplot(3, 3, 5)
    if 'transformer_prediction' in test_df.columns:
        plt.hist(test_df['transformer_prediction'], bins=30, alpha=0.7, color='orange')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Neutral')
        plt.title('🤖 Transformer Predictions Distribution')
        plt.xlabel('Prediction Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 6. 감정 지표 시계열
    plt.subplot(3, 3, 6)
    if 'positive_google_news' in test_df.columns and 'negative_google_news' in test_df.columns:
        # 날짜별 평균 감정 계산
        sentiment_by_date = test_df.groupby('date').agg({
            'positive_google_news': 'mean',
            'negative_google_news': 'mean'
        }).reset_index()
        
        plt.plot(sentiment_by_date['date'], sentiment_by_date['positive_google_news'], 
                'g-', label='Positive', alpha=0.7)
        plt.plot(sentiment_by_date['date'], sentiment_by_date['negative_google_news'], 
                'r-', label='Negative', alpha=0.7)
        plt.title('😊 Google News Sentiment Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    # 7. 특성 중요도 (근사치)
    plt.subplot(3, 3, 7)
    feature_groups = ['Tech', 'Macro', 'Transformer', 'Sentiment']
    feature_counts = [len(tech_indicators), len(macro_indicators), 
                     len(transformer_indicators), 
                     len(sentiment_indicators)]
    
    print(f"feature_groups 길이: {len(feature_groups)}")
    print(f"feature_counts 길이: {len(feature_counts)}")
    print(f"feature_groups: {feature_groups}")
    print(f"feature_counts: {feature_counts}")
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    plt.bar(feature_groups, feature_counts, color=colors, alpha=0.7)
    plt.title('📊 Feature Groups Distribution')
    plt.xlabel('Feature Groups')
    plt.ylabel('Number of Features')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 8. 수익률 분포
    plt.subplot(3, 3, 8)
    daily_returns = df_account_value["account_value"].pct_change().dropna() * 100
    plt.hist(daily_returns, bins=50, alpha=0.7, color='skyblue')
    plt.axvline(x=daily_returns.mean(), color='red', linestyle='--', 
                label=f'Mean: {daily_returns.mean():.2f}%')
    plt.title('📈 Daily Returns Distribution')
    plt.xlabel('Daily Return (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. 성과 요약
    plt.subplot(3, 3, 9)
    plt.axis('off')
    summary_text = f"""
    📊 Enhanced Model Summary
    
    🎯 Total Return: {total_return:.1%}
    📈 CAGR: {cagr:.1%}
    📉 Max Drawdown: {mdd:.1%}
    ⚡ Sharpe Ratio: {sharpe:.2f}
    📊 Volatility: {volatility:.1%}
    
    🆚 vs Benchmark: {outperformance:.1%}
    
    🔧 Total Features: {len(ALL_INDICATORS)}
    🤖 With Transformer: ✓
    😊 With Sentiment: ✓
    💼 With Financials: ✓
    """
    plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig("results/enhanced_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 확장된 분석 결과가 'results/enhanced_comprehensive_analysis.png'로 저장되었습니다!")
    
    # =============================================================================
    # 모델 및 결과 저장
    # =============================================================================
    
    # 결과 데이터프레임 저장
    result_summary = pd.DataFrame({
        'metric': ['Total Return', 'CAGR', 'Max Drawdown', 'Sharpe Ratio', 'Volatility', 'Outperformance'],
        'value': [total_return, cagr, mdd, sharpe, volatility, outperformance]
    })
    
    result_summary.to_csv("results/enhanced_performance_summary.csv", index=False)
    
    # 포트폴리오 가치 저장
    df_account_value.to_csv("results/enhanced_portfolio_values.csv", index=False)
    
    # 거래 기록 저장
    if len(df_actions) > 0:
        df_actions.to_csv("results/enhanced_trading_actions.csv", index=False)
    
    print(f"\n💾 추가 결과 파일들이 저장되었습니다:")
    print(f"   - results/enhanced_performance_summary.csv")
    print(f"   - results/enhanced_portfolio_values.csv")
    print(f"   - results/enhanced_trading_actions.csv")
    
    print(f"\n" + "="*60)
    print(f"✅ 확장된 FinRL 시스템 실행 완료!")
    print(f"="*60)
    
    return {
        'trained_model': trained_model,
        'performance': {
            'total_return': total_return,
            'cagr': cagr,
            'mdd': mdd,
            'sharpe': sharpe,
            'volatility': volatility,
            'outperformance': outperformance
        },
        'data_processor': data_processor,
        'feature_stats': {
            'total_features': len(ALL_INDICATORS),
            'tech_features': len(tech_indicators),
            'macro_features': len(macro_indicators),
            'transformer_features': len(transformer_indicators),
            'sentiment_features': len(sentiment_indicators)
        }
    }

if __name__ == "__main__":
    results = main()