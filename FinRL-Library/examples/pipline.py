import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any, Optional
import os
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

MAX_HOLDING_DAYS = 100

class TransformerClassifier(nn.Module):
    def __init__(self,input_dim,d_model=64,nhead=4, layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed = nn.Linear(input_dim,d_model)
        enc = nn.TransformerEncoderLayer(d_model,nhead,batch_first=True)
        self.encoder = nn.TransformerEncoder(enc,layers)
        self.head = nn.Linear(d_model,1)
    def forward(self,x):
        x = self.embed(x)
        x = self.encoder(x)
        return torch.sigmoid(self.head(x[:,-1]))  # (B,1)

class HierarchicalTradingEnvironment:
    """
    계층적 거래 환경: RL 에이전트가 Transformer와 감정 분석 점수를 참고
    """
    def __init__(self, transformer_model_path, market_data, sentiment_data_paths):
        # 모델 로드
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transformer_model = self._load_transformer_model(transformer_model_path)
        
        # 데이터 전처리 및 로드
        self.market_data = self._preprocess_market_data(market_data)
        self.sentiment_data = self._load_sentiment_data(sentiment_data_paths)
        
        # 피처 정의
        self.feature_cols = self._define_feature_columns()
        
        # 데이터 컬럼 확인 및 로깅
        print(f"시장 데이터 컬럼: {self.market_data.columns.tolist()}")
        if not self.sentiment_data.empty:
            print(f"감정 데이터 컬럼: {self.sentiment_data.columns.tolist()}")
        
        # 상태 공간 정의
        self.base_features = len(self.feature_cols)  # 기본 시장 데이터
        self.transformer_features = 3  # Transformer 예측 관련
        self.sentiment_features = 6   # 감정 분석 관련 (Google News + Reddit)
        self.state_dim = self.base_features + self.transformer_features + self.sentiment_features
        
        # 스케일러 초기화
        self.scaler = StandardScaler()
        self._fit_scaler()
        
        print(f"환경 초기화 완료: 상태 차원 = {self.state_dim}")
        print(f"- 기본 피처: {self.base_features}개")
        print(f"- Transformer 출력: {self.transformer_features}개 피처")
        print(f"- 감정 분석: {self.sentiment_features}개 피처")
        if not self.sentiment_data.empty:
            print(f"감정 데이터 기간: {self.sentiment_data['date'].min()} ~ {self.sentiment_data['date'].max()}")
   
    def _preprocess_market_data(self, market_data):
        """시장 데이터 전처리 (원본 코드와 동일하게)"""
        # 1. 기본 momentum 데이터 준비 (ROC_10, RSI_14, MACD 포함)
        momentum_data = market_data[['Date', 'Ticker', 'Close', 'ROC_10', 'RSI_14', 'MACD']].copy()
        momentum_data['Date'] = pd.to_datetime(momentum_data['Date'])
        momentum_data = momentum_data[momentum_data['Date'] >= '2020-01-01'].reset_index(drop=True)
        
        # 2. 거시경제 지표 로드 및 일간 데이터로 변환
        try:
            macro_path = "/Users/gamjawon/prometheus-11team/DATA/technical/macro_indicators_2020_2025-03.csv"
            if os.path.exists(macro_path):
                macro_data = pd.read_csv(macro_path)
                macro_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
                macro_data['Date'] = pd.to_datetime(macro_data['Date'])
                
                # 일간 인덱스로 변환
                macro_data.set_index('Date', inplace=True)
                daily_index = pd.date_range(start='2020-01-01', end='2025-04-30', freq='D')
                macro_data_daily = macro_data.reindex(daily_index, method='ffill').reset_index().rename(columns={'index': 'Date'})
            else:
                print("거시경제 데이터가 없어 기본값으로 설정")
                macro_data_daily = pd.DataFrame({'Date': momentum_data['Date'].unique()})
                for col in ['federal_funds_rate', 'treasury_yield', 'cpi', 'unemployment_rate']:
                    macro_data_daily[col] = 0.0
        except Exception as e:
            print(f"거시경제 데이터 로드 실패: {e}")
            macro_data_daily = pd.DataFrame({'Date': momentum_data['Date'].unique()})
            for col in ['federal_funds_rate', 'treasury_yield', 'cpi', 'unemployment_rate']:
                macro_data_daily[col] = 0.0
        
        # 3. 재무 데이터 로드 및 일간 데이터로 변환
        try:
            financial_path = "/Users/gamjawon/prometheus-11team/FinRL-Library/examples/data/M7_financial_data_2020_2025.csv"
            if os.path.exists(financial_path):
                financial_data = pd.read_csv(financial_path)
                
                # 회사명-티커 매핑
                company_to_ticker = {
                    'Alphabet': 'GOOGL', 'Amazon': 'AMZN', 'Apple': 'AAPL',
                    'Meta': 'META', 'Microsoft': 'MSFT', 'Nvidia': 'NVDA', 'Tesla': 'TSLA'
                }
                financial_data['Ticker'] = financial_data['Company'].map(company_to_ticker)
                financial_data['Date'] = pd.to_datetime(financial_data['Release Date'])
                
                features = ['Operating Income', 'Net Income', 'EPS Diluted', 'Total Assets', 'Shareholders Equity']
                daily_index = pd.date_range(start='2020-01-01', end='2025-04-30', freq='D')
                
                daily_financial_list = []
                for ticker in financial_data['Ticker'].unique():
                    df_t = financial_data.loc[financial_data['Ticker'] == ticker, ['Date', 'Ticker'] + features].copy()
                    df_t = df_t.set_index('Date').sort_index()
                    df_t = df_t[~df_t.index.duplicated(keep='last')]
                    df_t = df_t.reindex(daily_index).ffill()
                    df_t['Ticker'] = ticker
                    df_t = df_t.reset_index().rename(columns={'index': 'Date'})
                    daily_financial_list.append(df_t)
                
                daily_financial_data = pd.concat(daily_financial_list, ignore_index=True)
            else:
                print("재무 데이터가 없어 기본값으로 설정")
                daily_financial_data = momentum_data[['Date', 'Ticker']].copy()
                for col in ['Operating Income', 'Net Income', 'EPS Diluted', 'Total Assets', 'Shareholders Equity']:
                    daily_financial_data[col] = 0.0
        except Exception as e:
            print(f"재무 데이터 로드 실패: {e}")
            daily_financial_data = momentum_data[['Date', 'Ticker']].copy()
            for col in ['Operating Income', 'Net Income', 'EPS Diluted', 'Total Assets', 'Shareholders Equity']:
                daily_financial_data[col] = 0.0
        
        # 4. 모든 데이터 병합
        merged_data = pd.merge(momentum_data, macro_data_daily, on='Date', how='left')
        full_merged = pd.merge(merged_data, daily_financial_data, on=['Date', 'Ticker'], how='left')
        
        # 5. Target 생성 및 정리 - 수정된 버전
        full_merged['Signal'] = ((full_merged['Close'].shift(-5) / full_merged['Close']) - 1 > 0).astype(int)

        # NaN 처리를 더 관대하게
        full_merged = full_merged.dropna(subset=['Close', 'ROC_10', 'RSI_14', 'MACD']).reset_index(drop=True)  # 핵심 컬럼만 체크
        full_merged = full_merged.fillna(0)  # 나머지 NaN은 0으로 채움

        print(f"전처리된 데이터 shape: {full_merged.shape}")
        print(f"최종 데이터 head:")
        print(full_merged.head())
        return full_merged
    
    def _define_feature_columns(self):
        """피처 컬럼 정의"""
        all_possible_features = [
            'ROC_10', 'RSI_14', 'MACD', 'federal_funds_rate', 'treasury_yield', 
            'cpi', 'unemployment_rate', 'Operating Income', 'Net Income', 
            'EPS Diluted', 'Total Assets', 'Shareholders Equity'
        ]
        
        # 실제 존재하는 컬럼만 사용
        feature_cols = [col for col in all_possible_features if col in self.market_data.columns]
        print(f"사용 가능한 피처: {feature_cols}")
        return feature_cols
    
    def _load_transformer_model(self, model_path):
        """Transformer 모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        try:
            model = TransformerClassifier(input_dim=17).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            
            print("✅ Transformer 모델 로드 완료")
            return model
                
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise e
    
    def _fit_scaler(self):
        """스케일러 피팅"""
        if len(self.market_data) > 0 and len(self.feature_cols) > 0:
            sample_data = self.market_data[self.feature_cols].dropna()
            if len(sample_data) > 0:
                self.scaler.fit(sample_data)
                print(f"✅ 스케일러 피팅 완료: {len(self.feature_cols)}개 피처")
    
    def get_transformer_prediction(self, market_features):
        """Transformer 모델 예측 (17개 피처 입력)"""
        try:
            if len(market_features) == 0:
                return {'prediction': 0.5, 'confidence': 0.0, 'signal': 0}
            
            # 17개 피처로 맞춤
            if len(market_features) < 17:
                market_features = np.pad(market_features, (0, 17 - len(market_features)), 'constant', constant_values=0)
            elif len(market_features) > 17:
                market_features = market_features[:17]
            
            # 텐서 변환 및 모델 예측
            input_tensor = torch.tensor(market_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prediction = self.transformer_model(input_tensor).cpu().numpy().flatten()[0]
                
                confidence = abs(prediction - 0.5) * 2
                signal = 1 if prediction > 0.5 else 0
                
                return {
                    'prediction': float(prediction),
                    'confidence': float(confidence),
                    'signal': signal
                }
            
        except Exception as e:
            print(f"❌ Transformer 예측 오류: {e}")
            raise SystemExit(f"Transformer 예측 실패: {e}")
    
    def _get_base_features(self, date, ticker):
        """기본 시장 데이터 추출"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # 해당 날짜의 시장 데이터 찾기
        market_row = self.market_data[
            (pd.to_datetime(self.market_data['Date']) == date) & 
            (self.market_data['Ticker'] == ticker)
        ]
        
        if len(market_row) == 0:
            # 가장 가까운 날짜의 데이터 사용
            ticker_data = self.market_data[self.market_data['Ticker'] == ticker]
            if len(ticker_data) > 0:
                ticker_data['Date'] = pd.to_datetime(ticker_data['Date'])
                closest_idx = (ticker_data['Date'] - date).abs().argsort().iloc[0]
                market_row = ticker_data.iloc[closest_idx:closest_idx+1]
        
        if len(market_row) > 0 and self.feature_cols:
            features = market_row[self.feature_cols].values[0]
            features = np.nan_to_num(features, nan=0.0)
            features = np.clip(features, -1000, 1000)
            return features
        else:
            print(f"경고: 해당 날짜의 데이터가 없습니다 ({date}, {ticker})")
            return np.zeros(len(self.feature_cols) if self.feature_cols else 17)
        
    
    def get_state(self, date, ticker):
        """확장된 상태 벡터 생성"""
        try:
            # 1. 기본 시장 데이터 추출
            base_features = self._get_base_features(date, ticker)
            
            # 2. Transformer 예측
            transformer_result = self.get_transformer_prediction(base_features)
            transformer_features = np.array([
                transformer_result['prediction'],
                transformer_result['confidence'],
                transformer_result['signal']
            ])
            
            # 3. 감정 분석 피처
            sentiment_features = self.get_sentiment_features(date, ticker)
            
            # 4. 전체 상태 벡터 결합
            full_state = np.concatenate([
                base_features,
                transformer_features,
                sentiment_features
            ])
            
            # 5. 정규화 (기본 피처만)
            try:
                if len(self.feature_cols) > 0 and len(base_features) >= len(self.feature_cols):
                    normalized_base = self.scaler.transform(base_features[:len(self.feature_cols)].reshape(1, -1)).flatten()
                    
                    if len(base_features) > len(self.feature_cols):
                        remaining_base = base_features[len(self.feature_cols):]
                        normalized_base = np.concatenate([normalized_base, remaining_base])
                    
                    full_state = np.concatenate([
                        normalized_base,
                        transformer_features,
                        sentiment_features
                    ])
            except:
                pass  # 정규화 실패시 원본 사용
            
            return full_state
            
        except SystemExit:
            raise
        except Exception as e:
            print(f"❌ 상태 생성 오류 ({date}, {ticker}): {e}")
            raise SystemExit(f"상태 생성 실패: {e}")
    
    def _load_sentiment_data(self, sentiment_data_paths):
        """감정 분석 데이터 로드 및 병합"""
        sentiment_dfs = []
        
        for path in sentiment_data_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                
                # 데이터 정리
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                elif 'datadate' in df.columns:
                    df['date'] = pd.to_datetime(df['datadate'])
                
                # sentiment_score 컬럼을 positive, negative, neutral로 변환
                if 'sentiment_score' in df.columns:
                    df['positive'] = df['sentiment_score'].apply(lambda x: max(0, x))
                    df['negative'] = df['sentiment_score'].apply(lambda x: max(0, -x))
                    df['neutral'] = df['sentiment_score'].apply(lambda x: 1 - abs(x) if abs(x) <= 1 else 0)
                elif not all(col in df.columns for col in ['positive', 'negative', 'neutral']):
                    df['positive'] = 0.0
                    df['negative'] = 0.0
                    df['neutral'] = 1.0
                
                # 파일명으로 소스 구분
                if 'googlenews' in path:
                    df['source'] = 'google_news'
                elif 'reddit' in path:
                    df['source'] = 'reddit'
                else:
                    df['source'] = 'unknown'
                
                sentiment_dfs.append(df)
                print(f"감정 데이터 로드: {path} - {len(df)} 행")
            else:
                print(f"경고: 파일을 찾을 수 없습니다: {path}")
        
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
                fill_value=0
            )
            
            # 컬럼명 평탄화
            sentiment_pivot.columns = ['_'.join(col).strip() for col in sentiment_pivot.columns]
            sentiment_pivot = sentiment_pivot.reset_index()
            
            print(f"최종 감정 데이터 컬럼: {sentiment_pivot.columns.tolist()}")
            return sentiment_pivot
        else:
            print("경고: 감정 데이터를 로드할 수 없습니다.")
            return pd.DataFrame()
        
    def get_sentiment_features(self, date, ticker):
        """감정 분석 피처 추출"""
        if self.sentiment_data.empty:
            return np.zeros(6)
        
        # 해당 날짜의 감정 데이터 찾기
        sentiment_row = self.sentiment_data[
            (self.sentiment_data['date'] == date) & 
            (self.sentiment_data['tic'] == ticker)
        ]
        
        if len(sentiment_row) == 0:
            # 가장 가까운 날짜의 데이터 사용
            ticker_data = self.sentiment_data[self.sentiment_data['tic'] == ticker]
            if len(ticker_data) > 0:
                closest_date = ticker_data['date'].iloc[
                    (ticker_data['date'] - pd.to_datetime(date)).abs().argsort()[:1]
                ].iloc[0]
                sentiment_row = ticker_data[ticker_data['date'] == closest_date]
        
        if len(sentiment_row) > 0:
            row = sentiment_row.iloc[0]
            
            # 감정 점수 추출 (Google News + Reddit)
            google_pos = row.get('positive_google_news', 0)
            google_neg = row.get('negative_google_news', 0) 
            google_neu = row.get('neutral_google_news', 0)
            
            reddit_pos = row.get('positive_reddit', 0)
            reddit_neg = row.get('negative_reddit', 0)
            reddit_neu = row.get('neutral_reddit', 0)
            
            # 만약 소스별 분리가 안된 경우
            if google_pos == 0 and google_neg == 0 and reddit_pos == 0 and reddit_neg == 0:
                google_pos = row.get('positive_google_news', row.get('positive_unknown', 0))
                google_neg = row.get('negative_google_news', row.get('negative_unknown', 0))
                google_neu = row.get('neutral_google_news', row.get('neutral_unknown', 0))
                
                reddit_pos = row.get('positive_reddit', google_pos)
                reddit_neg = row.get('negative_reddit', google_neg)
                reddit_neu = row.get('neutral_reddit', google_neu)
            
            features = [google_pos, google_neg, google_neu, reddit_pos, reddit_neg, reddit_neu]
            return np.array(features)
        else:
            return np.zeros(6)
    
    def step(self, action_vector, date, ticker):
        """
        환경 스텝 실행: action_vector는 [a_1, ..., a_7, holding_action]
        """
        # 보유기간 추출
        raw_holding = action_vector[-1]
        holding_period = int((raw_holding + 1) / 2 * (MAX_HOLDING_DAYS - 1)) + 1

        current_state = self.get_state(date, ticker)
        next_date = pd.to_datetime(date) + timedelta(days=holding_period)
        next_state = self.get_state(next_date, ticker)
        
        # 보상: holding_period 동안의 누적 수익률
        reward = self._calculate_holding_reward(date, next_date, ticker, action_vector[0])
        
        done = False
        return next_state, reward, done
    
    def _calculate_holding_reward(self, entry_date, exit_date, ticker, position_action):
        """보유기간 동안 누적 수익률 기반 보상 계산"""
        entry_price = self._get_price(entry_date, ticker)
        exit_price = self._get_price(exit_date, ticker)

        if entry_price > 0 and exit_price > 0:
            return (exit_price / entry_price - 1) * position_action * 10
        else:
            return 0.0
    
    def _calculate_return(self, current_date, next_date, ticker):
        """수익률 계산"""
        try:
            current_price = self._get_price(current_date, ticker)
            next_price = self._get_price(next_date, ticker)
            
            if current_price > 0 and next_price > 0:
                return (next_price - current_price) / current_price
            else:
                return 0.0
        except:
            return 0.0
    
    def _get_price(self, date, ticker):
        """특정 날짜의 주가 가져오기"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # 수정: 컬럼명을 대문자로 통일
        price_row = self.market_data[
            (pd.to_datetime(self.market_data['Date']) == date) & 
            (self.market_data['Ticker'] == ticker)
        ]
        
        if len(price_row) > 0:
            return price_row['Close'].iloc[0]  # 수정: 대문자로 통일
        else:
            return 0.0
    
    def _calculate_reward(self, action, market_return, state):
        """향상된 보상 함수"""
        base_reward = action * market_return * 10
        
        # 상태 인덱스 수정 (기본 피처 개수가 변수이므로)
        state_idx = len(self.feature_cols)
        transformer_pred = state[state_idx]      
        transformer_conf = state[state_idx + 1]  
        transformer_signal = state[state_idx + 2]
        
        # 감정 점수
        google_positive = state[state_idx + 3]
        google_negative = state[state_idx + 4]
        reddit_positive = state[state_idx + 6]
        reddit_negative = state[state_idx + 7]
        
        # 방향성 일치 보너스
        direction_bonus = 0
        if transformer_signal > 0.5 and action > 0:
            direction_bonus = 0.1 * transformer_conf
        elif transformer_signal < 0.5 and action < 0:
            direction_bonus = 0.1 * transformer_conf
        
        # 감정-행동 일치 보너스
        sentiment_bonus = 0
        google_sentiment = google_positive - google_negative
        if google_sentiment > 0.1 and action > 0:
            sentiment_bonus += 0.03
        elif google_sentiment < -0.1 and action < 0:
            sentiment_bonus += 0.03
        
        reddit_sentiment = reddit_positive - reddit_negative
        if reddit_sentiment > 0.1 and action > 0:
            sentiment_bonus += 0.02
        elif reddit_sentiment < -0.1 and action < 0:
            sentiment_bonus += 0.02
        
        # 최종 보상 계산
        enhanced_reward = (
            base_reward * (1 + transformer_conf * 0.2) + 
            direction_bonus + 
            sentiment_bonus
        )
        
        return enhanced_reward

class EnhancedTD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim  # 8 (7종목 + holding_period)
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)

        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer(100000)
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0
    
    def select_action(self, state, noise=0.1):
        """행동 선택"""
        state_tensor = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        if noise != 0:
            action = action + np.random.normal(0, noise, size=self.action_dim)
            action = action.clip(-self.max_action, self.max_action)
            
        return action
    
    def train(self, batch_size=256):
        """TD3 훈련"""
        if len(self.replay_buffer) < batch_size:
            return
        
        self.total_it += 1
        
        # 배치 샘플링
        state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # 타겟 정책 스무딩
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # 타겟 Q 값 계산
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * 0.99 * target_Q
        
        # 현재 Q 값
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
        
        # 크리틱 업데이트
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        
        # 지연된 정책 업데이트
        if self.total_it % self.policy_freq == 0:
            # 액터 손실
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            
            # 액터 업데이트
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 소프트 업데이트
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic_1, self.critic_1_target)
            self.soft_update(self.critic_2, self.critic_2_target)
    
    def soft_update(self, source, target):
        """소프트 업데이트"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + source_param.data * self.tau)

class Actor(nn.Module):
    """액터 네트워크"""
    
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    """크리틱 네트워크"""
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1

class ReplayBuffer:
    """리플레이 버퍼"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, state, action, next_state, reward, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, next_state, reward, done = zip(*[self.buffer[i] for i in batch])
        
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(next_state),
            torch.FloatTensor(reward),
            torch.FloatTensor(done)
        )
    
    def __len__(self):
        return len(self.buffer)

class HierarchicalTradingSystem:
    """계층적 거래 시스템 통합 클래스"""
    
    def __init__(self, transformer_model_path, market_data, sentiment_data_paths):
        # 환경 초기화
        self.env = HierarchicalTradingEnvironment(
            transformer_model_path, market_data, sentiment_data_paths
        )
        
        # RL 에이전트 초기화
        self.agent = EnhancedTD3Agent(
            state_dim=self.env.state_dim,
            action_dim=8,  # 8개 종목
            max_action=1.0  # 정규화된 행동 범위
        )
        
        # 성능 추적
        self.performance_log = []
        
    def train_integrated_system(self, episodes=1000):
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']

        try:
            for episode in range(episodes):
                total_reward = 0
                steps = 0
                train_dates = self.sample_training_period()

                if isinstance(train_dates, pd.Series):
                    train_dates = train_dates.tolist()
                elif hasattr(train_dates, 'tolist'):
                    train_dates = train_dates.tolist()

                if len(train_dates) < 2:
                    print(f"Episode {episode}: 훈련 날짜가 부족합니다. 건너뜁니다.")
                    continue

                for i in range(len(train_dates) - 1):
                    date = train_dates[i]
                    episode_rewards = []

                    for ticker_idx, ticker in enumerate(tickers):
                        try:
                            state = self.env.get_state(date, ticker)
                            action_vector = self.agent.select_action(state)

                            # 종목별 행동
                            pos_action = action_vector[ticker_idx]

                            # 환경 스텝 (보유기간 포함)
                            next_state, reward, done = self.env.step(action_vector, date, ticker)

                            self.agent.replay_buffer.add(
                                state, action_vector, next_state, reward, done
                            )

                            if len(self.agent.replay_buffer) > 1000:
                                self.agent.train()

                            total_reward += reward
                            episode_rewards.append(reward)
                            steps += 1

                        except Exception as e:
                            print(f"Episode {episode}, {ticker}, {date}: 오류 발생 - {e}")
                            continue

                    if len(episode_rewards) == 0:
                        break

                avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                self.performance_log.append({
                    'episode': episode,
                    'total_reward': total_reward,
                    'avg_reward': avg_reward,
                    'steps': steps
                })

                if episode % 100 == 0:
                    print(f"Episode {episode}: Avg Reward = {avg_reward:.4f}, Total Reward = {total_reward:.4f}")

            print(f"✅ 훈련 완료: {episodes} 에피소드")
        except Exception as e:
            print(f"❌ 훈련 중 오류: {e}")
            raise SystemExit(f"훈련 실패: {e}")
            
    def make_trading_decision(self, date, ticker):
        """실제 거래 결정"""
        
        try:
            # 현재 상태 획득
            state = self.env.get_state(date, ticker)
            
            # RL 에이전트 행동 선택 (노이즈 제거)
            action = self.agent.select_action(state, noise=0)
            
            # 의사결정 해석
            decision = self.interpret_action(action[0], state)  # 첫 번째 행동만 사용
            
            return decision
            
        except SystemExit:
            # 시스템 종료 신호 재전파
            raise
        except Exception as e:
            print(f"❌ 거래 결정 오류 ({date}, {ticker}): {e}")
            print(f"프로그램을 종료합니다.")
            raise SystemExit(f"거래 결정 실패: {e}")
    
    def interpret_action(self, action, state):
        """행동을 거래 결정으로 해석"""
        
        # 상태 인덱스 수정 (기본 피처 개수가 변수이므로)
        state_idx = len(self.env.feature_cols)
        transformer_pred = state[state_idx]
        transformer_conf = state[state_idx + 1]
        transformer_signal = state[state_idx + 2]
        
        # 감정 점수
        google_sentiment = state[state_idx + 3] - state[state_idx + 4]  # positive - negative
        reddit_sentiment = state[state_idx + 6] - state[state_idx + 7]  # positive - negative
        
        # 행동 강도 계산
        action_strength = abs(action)
        
        # 의사결정 로직
        if action_strength < 0.1:
            decision = "HOLD"
            confidence = "LOW"
        elif action > 0:
            decision = "BUY"
            confidence = "HIGH" if action_strength > 0.5 else "MEDIUM"
        else:
            decision = "SELL"
            confidence = "HIGH" if action_strength > 0.5 else "MEDIUM"
        
        return {
            'action': decision,
            'confidence': confidence,
            'quantity': action_strength,
            'reasoning': {
                'transformer_prediction': f"{transformer_pred:.2f}",
                'transformer_confidence': f"{transformer_conf:.2f}",
                'transformer_signal': 'UP' if transformer_signal > 0.5 else 'DOWN',
                'google_sentiment': f"{google_sentiment:.2f}",
                'reddit_sentiment': f"{reddit_sentiment:.2f}",
                'rl_action': f"{action:.2f}"
            }
        }
    
    def sample_training_period(self):
        """훈련용 날짜 샘플링"""
        # 실제 데이터 기간에서 샘플링
        if len(self.env.market_data) > 0:
            available_dates = pd.to_datetime(self.env.market_data['Date']).unique()  # 수정: 대문자로 통일
            available_dates = pd.Series(available_dates).sort_values()
            
            # 30일 기간 선택
            if len(available_dates) >= 30:
                start_idx = np.random.randint(0, len(available_dates) - 30)
                selected_dates = available_dates.iloc[start_idx:start_idx + 30]
                return selected_dates.tolist()
            else:
                return available_dates.tolist()
        else:
            # 기본 기간
            start_date = pd.to_datetime('2020-01-01')
            date_range = pd.date_range(start_date, periods=30, freq='D')
            return date_range.tolist()

def load_market_data(file_path):
    """시장 데이터 로드"""
    try:
        data = pd.read_csv(file_path)
        
        # 컬럼명 통일
        if 'date' in data.columns:
            data = data.rename(columns={'date': 'Date'})
        if 'tic' in data.columns:
            data = data.rename(columns={'tic': 'Ticker'})
        if 'ticker' in data.columns:
            data = data.rename(columns={'ticker': 'Ticker'})
            
        print(f"시장 데이터 로드 완료: {file_path} - {len(data)} 행")
        print(f"컬럼: {data.columns.tolist()}")
        return data
    except Exception as e:
        print(f"시장 데이터 로드 실패: {e}")
        return pd.DataFrame()

def main():
    """메인 실행 함수 - 백테스트 코드 제거"""
    
    # 파일 경로 설정 
    transformer_model_path = '/Users/gamjawon/prometheus-11team/model/transformer_classifier_best.pt'
    market_data_path = "/Users/gamjawon/prometheus-11team/DATA/technical/M7_stock_momentum_data_fixed.csv"
    sentiment_data_paths = [
        "/Users/gamjawon/prometheus-11team/FinRL-Library/examples/data/M7_googlenews_2020_2022_sentiment_feature.csv",
        "/Users/gamjawon/prometheus-11team/FinRL-Library/examples/data/M7_reddits_2022_2025_sentiment_feature.csv"
    ]

    print("📊 데이터 및 모델 로딩 중...")

    # 기본 momentum 데이터 로드
    market_data_raw = load_market_data(market_data_path)
    
    if market_data_raw.empty:
        print("❌ 시장 데이터 로드 실패")
        return
    
    # 필요한 컬럼만 선택하고 날짜 필터링
    required_cols = ['Date', 'Ticker', 'Close', 'ROC_10', 'RSI_14', 'MACD']
    market_data_filtered = market_data_raw[required_cols].copy()
    market_data_filtered['Date'] = pd.to_datetime(market_data_filtered['Date'])
    market_data_filtered = market_data_filtered[market_data_filtered['Date'] >= '2020-01-01'].reset_index(drop=True)

    print(f"✅ 기본 시장 데이터 준비 완료: {len(market_data_filtered)} rows")
    
    # Transformer 모델 경로 확인
    if not os.path.exists(transformer_model_path):
        raise FileNotFoundError(f"Transformer 모델을 찾을 수 없습니다: {transformer_model_path}")
    
    try:
        # 통합 시스템 초기화 (전처리 포함)
        trading_system = HierarchicalTradingSystem(
            transformer_model_path, market_data_filtered, sentiment_data_paths
        )
        
        print(f"✅ 시스템 초기화 완료 - 상태 차원: {trading_system.env.state_dim}")
        
        # 훈련/테스트 데이터 분리 (전처리된 데이터에서)
        split_date = '2024-01-01'
        full_data = trading_system.env.market_data
        
        train_data = full_data[full_data['Date'] < split_date].copy().reset_index(drop=True)
        test_data = full_data[full_data['Date'] >= split_date].copy().reset_index(drop=True)
        
        print(f"✅ 훈련 데이터: {train_data['Date'].min().date()} ~ {train_data['Date'].max().date()} ({len(train_data)} rows)")
        print(f"✅ 테스트 데이터: {test_data['Date'].min().date()} ~ {test_data['Date'].max().date()} ({len(test_data)} rows)")
        
        # 시스템 훈련
        print("\n🚀 통합 시스템 훈련 시작...")
        # 훈련을 위해 market_data를 train_data로 일시 변경
        original_market_data = trading_system.env.market_data
        trading_system.env.market_data = train_data
        
        trading_system.train_integrated_system(episodes=500)
        
        # 원본 데이터로 복원
        trading_system.env.market_data = original_market_data
        
        # 실제 거래 결정 테스트
        print("\n📊 거래 결정 테스트...")
        
        # 테스트할 날짜 (테스트 데이터에서 사용 가능한 날짜 선택)
        available_dates = pd.to_datetime(test_data['Date']).unique()
        available_dates = sorted(available_dates)
        test_date = available_dates[10] if len(available_dates) >= 10 else available_dates[0]
        
        test_tickers = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN']
        
        for ticker in test_tickers:
            try:
                decision = trading_system.make_trading_decision(test_date, ticker)
                print(f"\n{ticker} 거래 결정 ({test_date.strftime('%Y-%m-%d')}):")
                print(f"  📈 행동: {decision['action']}")
                print(f"  🎯 확신도: {decision['confidence']}")
                print(f"  📊 수량: {decision['quantity']:.3f}")
                print(f"  🧠 근거:")
                for key, value in decision['reasoning'].items():
                    print(f"    - {key}: {value}")
                    
            except Exception as e:
                print(f"  ❌ {ticker} 결정 생성 실패: {e}")
        
        # 성능 요약
        if trading_system.performance_log:
            print("\n📈 훈련 성능 요약:")
            recent_performance = trading_system.performance_log[-50:]  # 최근 50 에피소드
            avg_reward = np.mean([p['avg_reward'] for p in recent_performance])
            print(f"  평균 보상 (최근 50 에피소드): {avg_reward:.4f}")
            print(f"  총 훈련 에피소드: {len(trading_system.performance_log)}")
        
        # 모델 저장
        save_trained_model(trading_system, "trained_hierarchical_model.pth")
        
        print("\n✅ 시스템 실행 완료!")
        
    except Exception as e:
        print(f"❌ 시스템 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def save_trained_model(trading_system, save_path):
    """훈련된 모델 저장"""
    try:
        model_state = {
            'actor_state_dict': trading_system.agent.actor.state_dict(),
            'critic_1_state_dict': trading_system.agent.critic_1.state_dict(),
            'critic_2_state_dict': trading_system.agent.critic_2.state_dict(),
            'state_dim': trading_system.env.state_dim,
            'action_dim': trading_system.agent.action_dim,
            'performance_log': trading_system.performance_log
        }
        
        torch.save(model_state, save_path)
        print(f"✅ 훈련된 모델 저장: {save_path}")
        
    except Exception as e:
        print(f"❌ 모델 저장 실패: {e}")

def load_trained_model(load_path, transformer_model_path, market_data, sentiment_data_paths, weights_only=False):
    """저장된 모델 로드"""
    try:
        # 시스템 초기화
        trading_system = HierarchicalTradingSystem(
            transformer_model_path, market_data, sentiment_data_paths
        )
        
        # 저장된 상태 로드
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=weights_only)
        
        trading_system.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        trading_system.agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        trading_system.agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        
        if 'performance_log' in checkpoint:
            trading_system.performance_log = checkpoint['performance_log']
        
        print(f"✅ 훈련된 모델 로드: {load_path}")
        return trading_system
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None

if __name__ == "__main__":
    main()