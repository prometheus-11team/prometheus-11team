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

class TransformerClassifier(nn.Module):
    """Transformer 기반 분류 모델 (로드용)"""
    
    def __init__(self, input_dim, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers
        
        self.embed = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 1)
        
        # 예측 확률 메서드 추가
        self.predict_proba = self._predict_proba
    
    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return torch.sigmoid(self.head(x[:, -1]))
    
    def _predict_proba(self, x):
        """예측 확률 반환 (2클래스 분류용)"""
        with torch.no_grad():
            self.eval()
            prob_positive = self.forward(x).cpu().numpy()
            prob_negative = 1 - prob_positive
            return np.column_stack([prob_negative, prob_positive])

class HierarchicalTradingEnvironment:
    """
    계층적 거래 환경: RL 에이전트가 Transformer와 감정 분석 점수를 참고
    """
    
    def __init__(self, transformer_model_path, market_data, sentiment_data_paths):
        # 모델 로드
        self.transformer_model = self._load_transformer_model(transformer_model_path)
        
        # 데이터 로드
        self.market_data = market_data
        self.sentiment_data = self._load_sentiment_data(sentiment_data_paths)
        
        # 데이터 컬럼 확인 및 로깅
        print(f"시장 데이터 컬럼: {self.market_data.columns.tolist()}")
        if not self.sentiment_data.empty:
            print(f"감정 데이터 컬럼: {self.sentiment_data.columns.tolist()}")
        
        # 상태 공간 정의
        self.base_features = 12  # Transformer 입력용 (기본 시장 데이터)
        self.transformer_features = 3  # Transformer 예측 관련
        self.sentiment_features = 6   # 감정 분석 관련 (Google News + Reddit)
        self.state_dim = self.base_features + self.transformer_features + self.sentiment_features
        
        # 스케일러 초기화
        self.scaler = StandardScaler()
        self._fit_scaler()
        
        print(f"환경 초기화 완료: 상태 차원 = {self.state_dim}")
        print(f"- Transformer 입력: {self.base_features}개 피처")
        print(f"- Transformer 출력: {self.transformer_features}개 피처")
        print(f"- 감정 분석: {self.sentiment_features}개 피처")
        if not self.sentiment_data.empty:
            print(f"감정 데이터 기간: {self.sentiment_data['date'].min()} ~ {self.sentiment_data['date'].max()}")
    
    def _load_transformer_model(self, model_path):
        """Transformer 모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        try:
            # PyTorch 2.6+ 호환성을 위해 weights_only=False 설정
            loaded_obj = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 완전한 모델 객체인 경우
            if hasattr(loaded_obj, 'eval'):
                print("✅ 완전한 모델 객체 로드됨")
                loaded_obj.eval()
                return loaded_obj
            
            # state_dict나 OrderedDict인 경우
            elif isinstance(loaded_obj, (dict, torch.nn.modules.container.OrderedDict)):
                print("❌ State dict 형태의 모델은 지원하지 않습니다.")
                print("완전한 모델 객체로 저장된 파일을 사용해주세요.")
                raise ValueError("State dict 형태의 모델은 지원하지 않습니다.")
            
            else:
                print(f"❌ 알 수 없는 모델 형태: {type(loaded_obj)}")
                raise ValueError(f"지원하지 않는 모델 형태: {type(loaded_obj)}")
                
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise e
        
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
                
                # 컬럼명 확인 및 통일
                print(f"감정 데이터 컬럼: {df.columns.tolist()}")
                
                # sentiment_score 컬럼을 positive, negative, neutral로 변환
                if 'sentiment_score' in df.columns:
                    # sentiment_score를 기반으로 positive/negative/neutral 생성
                    df['positive'] = df['sentiment_score'].apply(lambda x: max(0, x))
                    df['negative'] = df['sentiment_score'].apply(lambda x: max(0, -x))
                    df['neutral'] = df['sentiment_score'].apply(lambda x: 1 - abs(x) if abs(x) <= 1 else 0)
                elif not all(col in df.columns for col in ['positive', 'negative', 'neutral']):
                    # 필요한 컬럼이 없으면 기본값으로 채우기
                    df['positive'] = 0.0
                    df['negative'] = 0.0
                    df['neutral'] = 1.0
                    print(f"경고: 감정 점수 컬럼이 없어 기본값으로 설정")
                
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
            # 데이터 병합
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
    
    def _fit_scaler(self):
        """스케일러 피팅"""
        if len(self.market_data) > 0:
            # 기본 피처들로 스케일러 피팅
            base_features = ['close', 'roc_10', 'rsi_14', 'macd', 
                           'federal_funds_rate', 'treasury_yield', 
                           'cpi', 'unemployment_rate']
            
            available_features = [f for f in base_features if f in self.market_data.columns]
            if available_features:
                sample_data = self.market_data[available_features].dropna()
                if len(sample_data) > 0:
                    self.scaler.fit(sample_data)
    
    def get_transformer_prediction(self, market_features):
        """Transformer 모델 예측 (12개 피처 입력)"""
        try:
            
            # 입력 데이터 검증
            if len(market_features) == 0:
                print("경고: 입력 피처가 비어있습니다.")
                return {
                    'prediction': 0.5,
                    'confidence': 0.0,
                    'signal': 0
                }
            
            # 입력 데이터 준비 (12개 피처)
            if len(market_features) < 12:
                # 부족한 피처는 0으로 채움
                market_features = np.pad(market_features, (0, 12 - len(market_features)), 'constant', constant_values=0)
            
            # 정확히 12개 피처만 사용
            input_features = market_features[:12]
            
            # 모델의 입력 차원 확인
            expected_input_dim = self.transformer_model.input_dim
            actual_input_dim = len(input_features)
            
            if expected_input_dim != actual_input_dim:
                
                # 긴급 대응: 차원 맞춤
                if expected_input_dim < actual_input_dim:
                    input_features = input_features[:expected_input_dim]
                elif expected_input_dim > actual_input_dim:
                    input_features = np.pad(input_features, (0, expected_input_dim - actual_input_dim), 'constant', constant_values=0)
            
            # 텐서 변환
            input_tensor = torch.FloatTensor(input_features).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                # 단계별 실행으로 어디서 오류가 나는지 확인
                try:
                    # 1. 임베딩 레이어
                    embedded = self.transformer_model.embed(input_tensor)
                    
                    # 2. 인코더
                    encoded = self.transformer_model.encoder(embedded)
                    
                    # 3. 헤드 (마지막 레이어)
                    head_input = encoded[:, -1]  # 마지막 시퀀스 위치
                    
                    output = self.transformer_model.head(head_input)
                    
                    # 시그모이드 적용
                    prediction = torch.sigmoid(output).item()
                    
                except Exception as layer_error:
                    print(f"❌ 모델 레이어 실행 오류: {layer_error}")
                    raise layer_error
                
                # 예측 결과 처리
                confidence = abs(prediction - 0.5) * 2
                
                return {
                    'prediction': float(prediction),
                    'confidence': float(confidence),
                    'signal': 1 if prediction > 0.5 else 0
                }
            
        except Exception as e:
            print(f"❌ Transformer 예측 오류: {e}")
            print(f"프로그램을 종료합니다.")
            raise SystemExit(f"Transformer 예측 실패: {e}")
    
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
            features = []
            
            # Google News 감정 점수 (컬럼명 유연하게 처리)
            google_pos = row.get('positive_google_news', 0)
            google_neg = row.get('negative_google_news', 0) 
            google_neu = row.get('neutral_google_news', 0)
            
            # Reddit 감정 점수 (컬럼명 유연하게 처리)
            reddit_pos = row.get('positive_reddit', 0)
            reddit_neg = row.get('negative_reddit', 0)
            reddit_neu = row.get('neutral_reddit', 0)
            
            # 만약 소스별 분리가 안된 경우, 전체 감정 점수 사용
            if google_pos == 0 and google_neg == 0 and reddit_pos == 0 and reddit_neg == 0:
                # 사용 가능한 감정 점수 컬럼 찾기
                available_cols = [col for col in row.index if 'positive' in col or 'negative' in col or 'neutral' in col]
                if available_cols:
                    # 첫 번째 소스의 감정 점수를 Google News로 사용
                    google_pos = row.get('positive_google_news', row.get('positive_unknown', 0))
                    google_neg = row.get('negative_google_news', row.get('negative_unknown', 0))
                    google_neu = row.get('neutral_google_news', row.get('neutral_unknown', 0))
                    
                    # 두 번째 소스나 동일한 값을 Reddit으로 사용
                    reddit_pos = row.get('positive_reddit', google_pos)
                    reddit_neg = row.get('negative_reddit', google_neg)
                    reddit_neu = row.get('neutral_reddit', google_neu)
            
            features = [google_pos, google_neg, google_neu, reddit_pos, reddit_neg, reddit_neu]
            
            return np.array(features)
        else:
            return np.zeros(6)
    
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
            
            # 5. 정규화
            try:
                # 기본 피처만 정규화 (추가 피처는 이미 0-1 범위)
                normalized_base = self.scaler.transform(base_features.reshape(1, -1)).flatten()
                full_state = np.concatenate([
                    normalized_base,
                    transformer_features,
                    sentiment_features
                ])
            except:
                pass  # 정규화 실패시 원본 사용
            
            return full_state
            
        except SystemExit:
            # Transformer 오류로 인한 시스템 종료
            raise
        except Exception as e:
            print(f"❌ 상태 생성 오류 ({date}, {ticker}): {e}")
            print(f"프로그램을 종료합니다.")
            raise SystemExit(f"상태 생성 실패: {e}")
    
    def _get_base_features(self, date, ticker):
        """기본 시장 데이터 추출 (Transformer용 12개 피처)"""
        # 날짜 형식 통일
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # 컬럼명 확인 (tic 또는 ticker)
        ticker_col = 'tic' if 'tic' in self.market_data.columns else 'ticker'
        
        # 해당 날짜의 시장 데이터 찾기
        market_row = self.market_data[
            (pd.to_datetime(self.market_data['date']) == date) & 
            (self.market_data[ticker_col] == ticker)
        ]
        
        if len(market_row) == 0:
            # 가장 가까운 날짜의 데이터 사용
            ticker_data = self.market_data[self.market_data[ticker_col] == ticker]
            if len(ticker_data) > 0:
                ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                closest_idx = (ticker_data['date'] - date).abs().argsort().iloc[0]
                market_row = ticker_data.iloc[closest_idx:closest_idx+1]
        
        if len(market_row) > 0:
            # 기본 피처 추출 (가능한 한 많은 컬럼 사용)
            primary_features = [
                'close', 'roc_10', 'rsi_14', 'macd', 
                'federal_funds_rate', 'treasury_yield', 'cpi', 'unemployment_rate'
            ]
            
            # 추가 피처 (데이터에 있을 수 있는 다른 컬럼들)
            additional_features = [
                'open', 'high', 'low', 'volume', 'vwap', 'day_of_week', 
                'month', 'quarter', 'market_cap', 'pe_ratio', 'pb_ratio',
                'dividend_yield', 'beta', 'volatility', 'momentum_1m', 'momentum_3m'
            ]
            
            # 전체 피처 리스트
            all_features = primary_features + additional_features
            
            # 실제 존재하는 컬럼만 사용
            available_columns = [col for col in all_features if col in market_row.columns]
            
            if available_columns:
                features = market_row[available_columns].values[0]
                
                # NaN 값 처리
                features = np.nan_to_num(features, nan=0.0)
                
                
                # 12개 피처로 맞춤
                if len(features) < 12:
                    # 부족한 피처는 0으로 채움
                    features = np.pad(features, (0, 12 - len(features)), 'constant', constant_values=0)
                elif len(features) > 12:
                    # 넘치는 피처는 자름
                    features = features[:12]
                
                # 기본 정규화 (너무 큰 값 방지)
                features = np.clip(features, -1000, 1000)
                
                return features
            else:
                print(f"경고: 사용 가능한 피처 컬럼이 없습니다.")
                print(f"시장 데이터 컬럼: {market_row.columns.tolist()}")
                return np.zeros(12)
        else:
            print(f"경고: 해당 날짜의 데이터가 없습니다 ({date}, {ticker})")
            return np.zeros(12)
    
    def step(self, action, date, ticker):
        """환경 스텝 실행"""
        # 현재 상태
        current_state = self.get_state(date, ticker)
        
        # 다음 날짜 계산
        next_date = pd.to_datetime(date) + timedelta(days=1)
        
        # 다음 상태
        next_state = self.get_state(next_date, ticker)
        
        # 수익률 계산
        market_return = self._calculate_return(date, next_date, ticker)
        
        # 보상 계산
        reward = self._calculate_reward(action, market_return, current_state)
        
        # 종료 조건
        done = False  # 실제 구현에서는 에피소드 종료 조건 추가
        
        return next_state, reward, done
    
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
        
        # 컬럼명 확인 (tic 또는 ticker)
        ticker_col = 'tic' if 'tic' in self.market_data.columns else 'ticker'
        
        price_row = self.market_data[
            (pd.to_datetime(self.market_data['date']) == date) & 
            (self.market_data[ticker_col] == ticker)
        ]
        
        if len(price_row) > 0:
            return price_row['close'].iloc[0]
        else:
            return 0.0
    
    def _calculate_reward(self, action, market_return, state):
        """향상된 보상 함수"""
        # 기본 수익률 보상
        base_reward = action * market_return
        
        # 상태에서 예측 정보 추출
        transformer_pred = state[12]      # Transformer 예측 (12번째 인덱스)
        transformer_conf = state[13]      # Transformer 확신도 (13번째 인덱스)
        transformer_signal = state[14]    # Transformer 신호 (14번째 인덱스)
        
        # 감정 점수 (Google News + Reddit) - 15번째 인덱스부터
        google_positive = state[15]
        google_negative = state[16]
        reddit_positive = state[18]
        reddit_negative = state[19]
        
        # 1. 방향성 일치 보너스
        direction_bonus = 0
        if transformer_signal > 0.5 and action > 0:  # 둘 다 상승
            direction_bonus = 0.1 * transformer_conf
        elif transformer_signal < 0.5 and action < 0:  # 둘 다 하락
            direction_bonus = 0.1 * transformer_conf
        
        # 2. 감정-행동 일치 보너스
        sentiment_bonus = 0
        
        # Google News 감정 기반
        google_sentiment = google_positive - google_negative
        if google_sentiment > 0.1 and action > 0:
            sentiment_bonus += 0.03
        elif google_sentiment < -0.1 and action < 0:
            sentiment_bonus += 0.03
        
        # Reddit 감정 기반
        reddit_sentiment = reddit_positive - reddit_negative
        if reddit_sentiment > 0.1 and action > 0:
            sentiment_bonus += 0.02
        elif reddit_sentiment < -0.1 and action < 0:
            sentiment_bonus += 0.02
        
        # 3. 확신도 기반 조정
        confidence_multiplier = transformer_conf
        
        # 최종 보상 계산
        enhanced_reward = (
            base_reward * (1 + confidence_multiplier * 0.2) + 
            direction_bonus + 
            sentiment_bonus
        )
        
        return enhanced_reward

class EnhancedTD3Agent:
    """
    향상된 TD3 에이전트
    """
    
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # 네트워크 초기화
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        
        # 타겟 네트워크
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        
        # 타겟 네트워크 초기화
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # 옵티마이저
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)
        
        # 리플레이 버퍼 (간단한 구현)
        self.replay_buffer = ReplayBuffer(100000)
        
        # TD3 하이퍼파라미터
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
        
        # 크리틱 손실
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
    """
    계층적 거래 시스템 통합 클래스
    """
    
    def __init__(self, transformer_model_path, market_data, sentiment_data_paths):
        # 환경 초기화
        self.env = HierarchicalTradingEnvironment(
            transformer_model_path, market_data, sentiment_data_paths
        )
        
        # RL 에이전트 초기화
        self.agent = EnhancedTD3Agent(
            state_dim=self.env.state_dim,
            action_dim=7,  # 7개 종목
            max_action=1.0  # 정규화된 행동 범위
        )
        
        # 성능 추적
        self.performance_log = []
        
    def train_integrated_system(self, episodes=1000):
        """통합 시스템 훈련"""
        
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
        
        try:
            for episode in range(episodes):
                total_reward = 0
                steps = 0
                
                # 훈련 기간 샘플링
                train_dates = self.sample_training_period()
                
                # pandas Series를 리스트로 변환
                if isinstance(train_dates, pd.Series):
                    train_dates = train_dates.tolist()
                elif hasattr(train_dates, 'tolist'):
                    train_dates = train_dates.tolist()
                
                # 날짜 리스트 확인
                if len(train_dates) < 2:
                    print(f"Episode {episode}: 훈련 날짜가 부족합니다. 건너뜁니다.")
                    continue
                
                for i in range(len(train_dates) - 1):  # 마지막 날 제외
                    date = train_dates[i]
                    next_date = train_dates[i + 1]
                    
                    # 각 종목별로 거래 결정
                    episode_rewards = []
                    
                    for ticker_idx, ticker in enumerate(tickers):
                        try:
                            # 현재 상태 획득
                            state = self.env.get_state(date, ticker)
                            
                            # 행동 선택
                            action = self.agent.select_action(state)
                            
                            # 환경 스텝
                            next_state, reward, done = self.env.step(action[ticker_idx], date, ticker)
                            
                            # 경험 저장
                            self.agent.replay_buffer.add(
                                state, action, next_state, reward, done
                            )
                            
                            # 학습
                            if len(self.agent.replay_buffer) > 1000:
                                self.agent.train()
                            
                            total_reward += reward
                            episode_rewards.append(reward)
                            steps += 1
                            
                        except SystemExit:
                            # 심각한 오류로 인한 시스템 종료
                            raise
                        except Exception as e:
                            print(f"Episode {episode}, {ticker}, {date}: 오류 발생 - {e}")
                            print(f"프로그램을 종료합니다.")
                            raise SystemExit(f"훈련 중 오류 발생: {e}")
                    
                    if len(episode_rewards) == 0:  # 모든 종목에서 오류 발생
                        break
                
                # 에피소드 성능 기록
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
            
        except SystemExit:
            # 시스템 종료 신호 재전파
            raise
        except Exception as e:
            print(f"❌ 훈련 중 예상치 못한 오류: {e}")
            print(f"프로그램을 종료합니다.")
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
        
        # 모델 예측 정보 추출 (12개 기본 피처 이후)
        transformer_pred = state[12]
        transformer_conf = state[13]
        transformer_signal = state[14]
        
        # 감정 점수 (15번째 인덱스부터)
        google_sentiment = state[15] - state[16]  # positive - negative
        reddit_sentiment = state[18] - state[19]  # positive - negative
        
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
            available_dates = pd.to_datetime(self.env.market_data['date']).unique()
            available_dates = pd.Series(available_dates).sort_values()
            
            # 30일 기간 선택
            if len(available_dates) >= 30:
                start_idx = np.random.randint(0, len(available_dates) - 30)
                selected_dates = available_dates.iloc[start_idx:start_idx + 30]
                return selected_dates.tolist()  # 리스트로 변환
            else:
                return available_dates.tolist()  # 리스트로 변환
        else:
            # 기본 기간
            start_date = pd.to_datetime('2020-01-01')
            date_range = pd.date_range(start_date, periods=30, freq='D')
            return date_range.tolist()  # 리스트로 변환

def load_market_data(file_path):
    """시장 데이터 로드"""
    try:
        data = pd.read_csv(file_path)
        print(f"시장 데이터 로드 완료: {file_path} - {len(data)} 행")
        return data
    except Exception as e:
        print(f"시장 데이터 로드 실패: {e}")
        return pd.DataFrame()

def main():
    """메인 실행 함수"""
    
    # 파일 경로 설정
    transformer_model_path = "/Users/gamjawon/FinRL-Library/examples/models/transformer_classifier.pt"
    market_data_path = "/Users/gamjawon/FinRL-Library/examples/data/merged.csv"
    sentiment_data_paths = [
        "/Users/gamjawon/FinRL-Library/examples/data/M7_googlenews_2020_2022_sentiment_feature.csv",
        "/Users/gamjawon/FinRL-Library/examples/data/M7_reddits_2022_2025_sentiment_feature.csv"
    ]

    print("📊 데이터 및 모델 로딩 중...")

    # 시장 데이터 전체 로드
    market_data_all = load_market_data(market_data_path)
    market_data_all['date'] = pd.to_datetime(market_data_all['date'])

    # 🔹 훈련/테스트 데이터 분리
    train_data = market_data_all[market_data_all['date'] < '2023-07-01'].copy()
    test_data = market_data_all[market_data_all['date'] >= '2023-07-01'].copy()

    print(f"✅ 훈련 데이터 기간: {train_data['date'].min().date()} ~ {train_data['date'].max().date()} ({len(train_data)} rows)")
    print(f"✅ 테스트 데이터 기간: {test_data['date'].min().date()} ~ {test_data['date'].max().date()} ({len(test_data)} rows)")
    
    # Transformer 모델 경로 확인
    if not os.path.exists(transformer_model_path):
        raise FileNotFoundError(f"Transformer 모델을 찾을 수 없습니다: {transformer_model_path}")
    
    try:
        # 통합 시스템 초기화
        trading_system = HierarchicalTradingSystem(
            transformer_model_path, train_data, sentiment_data_paths
        )
        
        print(f"✅ 시스템 초기화 완료 - 상태 차원: {trading_system.env.state_dim}")
        
        # 시스템 훈련
        print("\n🚀 통합 시스템 훈련 시작...")
        trading_system.train_integrated_system(episodes=500)
        
        # 실제 거래 결정 테스트
        print("\n📊 거래 결정 테스트...")
        
        # 테스트할 날짜 (데이터에서 사용 가능한 날짜 선택)
        available_dates = pd.to_datetime(train_data['date']).unique()
        available_dates = sorted(available_dates)
        test_date = available_dates[-10] if len(available_dates) >= 10 else available_dates[-1]
        
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
        
    except Exception as e:
        print(f"❌ 시스템 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def generate_sample_market_data():
    """샘플 시장 데이터 생성"""
    print("샘플 시장 데이터 생성 중...")
    
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    data = []
    
    for ticker in tickers:
        base_price = np.random.uniform(100, 300)
        
        for i, date in enumerate(dates):
            # 랜덤 워크로 가격 생성
            if i == 0:
                price = base_price
            else:
                price = max(1, price + np.random.normal(0, price * 0.02))
            
            row = {
                'date': date,
                'tic': ticker,
                'close': price,
                'roc_10': np.random.normal(0, 0.05),
                'rsi_14': np.random.uniform(20, 80),
                'macd': np.random.normal(0, 0.1),
                'federal_funds_rate': np.random.uniform(0, 5),
                'treasury_yield': np.random.uniform(1, 4),
                'cpi': np.random.uniform(1, 6),
                'unemployment_rate': np.random.uniform(3, 10)
            }
            data.append(row)
    
    return pd.DataFrame(data)

def create_sample_transformer_model(save_path):
    """샘플 Transformer 모델 생성"""
    print("샘플 Transformer 모델 생성 중...")
    
    class SampleTransformerClassifier(nn.Module):
        def __init__(self, input_dim=13, hidden_dim=64, num_classes=2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, num_classes)
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
        
        def predict_proba(self, x):
            with torch.no_grad():
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                return probs.numpy()
    
    # 모델 생성 및 저장
    model = SampleTransformerClassifier()
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 모델 저장
    torch.save(model, save_path)
    print(f"샘플 모델 저장: {save_path}")

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

def evaluate_model_performance(trading_system, test_data, test_period_days=30):
    """모델 성능 평가"""
    print("\n📊 모델 성능 평가 중...")
    
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # 테스트 기간 선택
    available_dates = pd.to_datetime(test_data['date']).unique()
    test_dates = available_dates[-test_period_days:] if len(available_dates) >= test_period_days else available_dates
    
    total_returns = []
    decisions_log = []
    
    for date in test_dates:
        daily_returns = []
        
        for ticker in tickers:
            try:
                # 거래 결정
                decision = trading_system.make_trading_decision(date, ticker)
                
                # 실제 수익률 계산 (다음 날 기준)
                current_price = trading_system.env._get_price(date, ticker)
                next_date = date + timedelta(days=1)
                next_price = trading_system.env._get_price(next_date, ticker)
                
                if current_price > 0 and next_price > 0:
                    market_return = (next_price - current_price) / current_price
                    
                    # 포지션에 따른 수익률
                    if decision['action'] == 'BUY':
                        position_return = market_return * decision['quantity']
                    elif decision['action'] == 'SELL':
                        position_return = -market_return * decision['quantity']
                    else:  # HOLD
                        position_return = 0
                    
                    daily_returns.append(position_return)
                    
                    decisions_log.append({
                        'date': date,
                        'ticker': ticker,
                        'decision': decision['action'],
                        'quantity': decision['quantity'],
                        'market_return': market_return,
                        'position_return': position_return
                    })
                    
            except Exception as e:
                print(f"평가 중 오류 ({ticker}, {date}): {e}")
        
        if daily_returns:
            total_returns.append(np.mean(daily_returns))
    
    # 성능 지표 계산
    if total_returns:
        total_return = np.sum(total_returns)
        avg_daily_return = np.mean(total_returns)
        volatility = np.std(total_returns)
        sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0
        
        print(f"📈 평가 결과:")
        print(f"  총 수익률: {total_return:.4f}")
        print(f"  평균 일일 수익률: {avg_daily_return:.4f}")
        print(f"  변동성: {volatility:.4f}")
        print(f"  샤프 비율: {sharpe_ratio:.4f}")
        print(f"  거래 결정 수: {len(decisions_log)}")
        
        return {
            'total_return': total_return,
            'avg_daily_return': avg_daily_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'decisions_log': decisions_log
        }
    else:
        print("❌ 성능 평가 데이터가 부족합니다.")
        return None

if __name__ == "__main__":
    main()