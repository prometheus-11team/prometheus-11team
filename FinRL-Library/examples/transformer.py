import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import pickle

# 데이터 로드 및 전처리
def load_and_process_data():
    """데이터 로드 및 전처리"""
    
    # 1. 주가 및 기술지표 데이터
    momentum_data_url = '/Users/gamjawon/FinRL-Library/examples/data/M7_stock_data_with_indicators.csv'
    momentum_data = pd.read_csv(momentum_data_url)
    momentum_data = momentum_data[['date', 'ticker', 'close', 'ROC_10', 'RSI_14', 'MACD']]
    momentum_data['date'] = pd.to_datetime(momentum_data['date'])
    momentum_data = momentum_data[momentum_data['date'] >= '2020-01-01'].reset_index(drop=True)
    print(f"주가 데이터: {momentum_data.shape}")
    print(f"주가 데이터 ticker 종류: {momentum_data['ticker'].unique()}")
    
    # 2. 거시경제지표 데이터
    macro_data_url = '/Users/gamjawon/FinRL-Library/examples/data/macro_indicators_2020_2024.csv'
    macro_data = pd.read_csv(macro_data_url)
    macro_data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    macro_data['date'] = pd.to_datetime(macro_data['date'])
    
    # 거시경제지표 일간 데이터로 변환
    macro_data.set_index('date', inplace=True)
    daily_index = pd.date_range(start='2020-01-01', end='2025-04-30', freq='D')
    macro_data_daily = macro_data.reindex(daily_index, method='ffill')
    macro_data_daily = macro_data_daily.reset_index().rename(columns={'index': 'date'})
    print(f"거시경제 데이터: {macro_data_daily.shape}")
    
    # 3. 재무제표 데이터
    financial_data_url = '/Users/gamjawon/FinRL-Library/examples/data/M7_financial_data_2020_2025.csv'
    financial_data = pd.read_csv(financial_data_url)
    print(f"재무 데이터 원본: {financial_data.shape}")
    print(f"재무 데이터 컬럼: {financial_data.columns.tolist()}")
    print(f"재무 데이터 샘플:\n{financial_data.head()}")
    
    # 회사명과 Ticker 매핑
    company_to_ticker = {
        'Alphabet': 'GOOGL',
        'Amazon': 'AMZN',
        'Apple': 'AAPL',
        'Meta': 'META',
        'Microsoft': 'MSFT',
        'Nvidia': 'NVDA',
        'Tesla': 'TSLA'
    }
    
    financial_data['ticker'] = financial_data['Company'].map(company_to_ticker)
    financial_data['date'] = pd.to_datetime(financial_data['Release Date'])
    print(f"재무 데이터 매핑 후 ticker: {financial_data['ticker'].unique()}")
    print(f"재무 데이터 날짜 범위: {financial_data['date'].min()} ~ {financial_data['date'].max()}")
    
    # 필요한 재무 피처 선택
    features = ['Operating Income', 'Net Income', 'EPS Diluted', 'Total Assets', 'Shareholders Equity']
    
    # 재무데이터 일간 변환
    daily_financial_list = []
    for ticker in financial_data['ticker'].unique():
        if pd.isna(ticker):  # NaN ticker 스킵
            continue
            
        df_t = (
            financial_data
            .loc[financial_data['ticker'] == ticker, ['date', 'ticker'] + features]
            .copy()
            .set_index('date')
            .sort_index()
        )
        print(f"{ticker} 재무 데이터: {df_t.shape}")
        
        if len(df_t) == 0:  # 데이터가 없으면 스킵
            print(f"{ticker} 재무 데이터 없음")
            continue
            
        df_t = df_t[~df_t.index.duplicated(keep='last')]
        df_t = df_t.reindex(daily_index).ffill().infer_objects(copy=False)
        df_t['ticker'] = ticker
        df_t = df_t.reset_index().rename(columns={'index': 'date'})
        daily_financial_list.append(df_t)
    
    if daily_financial_list:
        daily_financial_data = pd.concat(daily_financial_list, ignore_index=True)
        print(f"일간 재무 데이터: {daily_financial_data.shape}")
        print(f"일간 재무 데이터 NaN 개수:\n{daily_financial_data[features].isna().sum()}")
    else:
        print("❌ 재무 데이터 처리 실패")
        # 재무 데이터 없이 진행
        daily_financial_data = None
    
    # 4. 데이터 병합
    # 주가 + 거시경제 (재무데이터는 모든 값이 NaN이므로 제외)
    full_merged = pd.merge(momentum_data, macro_data_daily, on='date', how='left')
    print(f"최종 병합 데이터: {full_merged.shape}")
    
    return full_merged

def prepare_training_data(df):
    """학습 데이터 준비"""
    
    print(f"초기 데이터 형태: {df.shape}")
    print(f"컬럼명: {df.columns.tolist()}")
    
    # 타겟 생성: 다음 5일 수익률 > 0 → 1, else 0
    df['date'] = pd.to_datetime(df['date'])
    print(f"날짜 변환 후: {df.shape}")
    
    # 티커별로 시프트 적용
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    df['Signal'] = df.groupby('ticker')['close'].transform(
        lambda x: ((x.shift(-5) / x) - 1 > 0).astype(int)
    )
    print(f"Signal 생성 후: {df.shape}")
    print(f"Signal 값 분포:\n{df['Signal'].value_counts()}")
    
    # NaN이 많은 컬럼 제거 (임계값: 50% 이상 NaN)
    nan_threshold = 0.5
    nan_ratio = df.isna().sum() / len(df)
    high_nan_cols = nan_ratio[nan_ratio > nan_threshold].index.tolist()
    
    if high_nan_cols:
        print(f"NaN이 많은 컬럼 제거: {high_nan_cols}")
        df = df.drop(columns=high_nan_cols)
    
    # Signal NaN 제거
    print(f"NaN 개수 (Signal): {df['Signal'].isna().sum()}")
    df = df.dropna(subset=['Signal']).reset_index(drop=True)
    print(f"Signal NaN 제거 후: {df.shape}")
    
    # 나머지 NaN 값들을 0으로 채우기 (또는 평균값으로)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['date', 'Signal']:
            df[col] = df[col].fillna(df[col].mean())
    
    print(f"NaN 처리 후 데이터 형태: {df.shape}")
    print(f"남은 NaN 개수:\n{df.isna().sum().sum()}")
    
    if len(df) == 0:
        print("❌ 모든 데이터가 제거되었습니다!")
        return None, None, None, None
    
    # 시계열 분할
    split_date = '2024-01-01'
    train_df = df[df['date'] < split_date].copy().reset_index(drop=True)
    test_df = df[df['date'] >= split_date].copy().reset_index(drop=True)
    
    print(f"학습 데이터: {train_df.shape}")
    print(f"테스트 데이터: {test_df.shape}")
    
    if len(train_df) == 0:
        print("❌ 학습 데이터가 비어있습니다!")
        return None, None, None, None
    
    # 피처 컬럼 선택
    feature_cols = [c for c in train_df.columns 
                   if c not in ['date', 'ticker', 'close', 'Signal']]
    
    print(f"피처 컬럼: {feature_cols}")
    print(f"피처 데이터 형태: {train_df[feature_cols].shape}")
    
    # 정규화
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    if len(test_df) > 0:
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    return train_df, test_df, feature_cols, scaler

def create_sequence(df, feature_cols, window=30):
    """시계열 윈도우 생성"""
    X, y = [], []
    vals = df[feature_cols].values
    sigs = df['Signal'].values
    
    for i in range(len(df) - window):
        X.append(vals[i:i+window])
        y.append(sigs[i+window])
    
    return np.array(X), np.array(y)

class TransformerClassifier(nn.Module):
    """Transformer 기반 분류 모델"""
    
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

def train_model(X_train, y_train, epochs=50, batch_size=32, lr=1e-3, val_ratio=0.2):
    """모델 학습"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    # 데이터 준비
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    ds = TensorDataset(X_t, y_t)
    
    # Train/Val 분할
    n_val = int(len(ds) * val_ratio)
    n_trn = len(ds) - n_val
    trn_ds, val_ds = random_split(ds, [n_trn, n_val])
    
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # 모델 초기화
    model = TransformerClassifier(input_dim=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    
    # 학습 루프
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        trn_losses = []
        for xb, yb in trn_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            trn_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_losses.append(loss.item())
        
        avg_trn_loss = np.mean(trn_losses)
        avg_val_loss = np.mean(val_losses)
        
        # 최적 모델 저장 (전체 모델 객체 복사)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 모델 전체를 CPU로 복사
            best_model = TransformerClassifier(input_dim=X_train.shape[2])
            best_model.load_state_dict(model.state_dict())
            best_model.eval()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Train Loss: {avg_trn_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    return best_model

def save_model_and_scaler(model, scaler, feature_cols, save_dir='./models'):
    """모델과 스케일러 저장 (완전한 객체 형태로)"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. 완전한 모델 객체 저장
    model_path = os.path.join(save_dir, 'transformer_classifier.pt')
    torch.save(model, model_path)  # 완전한 객체 저장
    print(f"✅ 완전한 모델 객체 저장: {model_path}")
    
    # 2. 스케일러와 피처 정보 저장
    scaler_path = os.path.join(save_dir, 'scaler_and_features.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'feature_cols': feature_cols
        }, f)
    print(f"✅ 스케일러 저장: {scaler_path}")
    
    # 3. 저장된 모델 검증 (PyTorch 2.6+ 호환)
    try:
        loaded_model = torch.load(model_path, map_location='cpu', weights_only=False)
        if hasattr(loaded_model, 'eval'):
            print("✅ 모델 저장 검증 완료 - 완전한 객체 형태")
            print(f"모델 타입: {type(loaded_model)}")
            print(f"모델 구조: {loaded_model}")
        else:
            print("❌ 모델 저장 검증 실패 - 완전한 객체가 아님")
    except Exception as e:
        print(f"❌ 모델 저장 검증 실패: {e}")

def load_trained_model(model_path, scaler_path):
    """학습된 모델과 스케일러 로드"""
    
    try:
        # 1. 완전한 모델 객체 로드 (PyTorch 2.6+ 호환)
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        if not hasattr(model, 'eval'):
            raise ValueError("저장된 모델이 완전한 객체가 아닙니다.")
        
        model.eval()
        print(f"✅ 완전한 모델 객체 로드: {model_path}")
        print(f"모델 타입: {type(model)}")
        
        # 2. 스케일러와 피처 정보 로드
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
        
        scaler = scaler_data['scaler']
        feature_cols = scaler_data['feature_cols']
        print(f"✅ 스케일러 로드 완료: {scaler_path}")
        print(f"피처 개수: {len(feature_cols)}")
        
        return model, scaler, feature_cols
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None, None, None

def check_model_exists(save_dir='./models'):
    """학습된 모델 존재 여부 확인"""
    
    model_path = os.path.join(save_dir, 'transformer_classifier.pt')
    scaler_path = os.path.join(save_dir, 'scaler_and_features.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"✅ 기존 모델 발견: {model_path}")
        
        # 모델 타입 확인 (PyTorch 2.6+ 호환)
        try:
            loaded_model = torch.load(model_path, map_location='cpu', weights_only=False)
            if hasattr(loaded_model, 'eval'):
                print("✅ 완전한 객체 형태로 저장되어 있음")
                return model_path, scaler_path
            else:
                print("❌ State dict 형태로 저장되어 있음 - 새로 학습 필요")
                return None, None
        except Exception as e:
            print(f"❌ 모델 파일 검증 실패: {e}")
            return None, None
    else:
        print("❌ 기존 모델 없음. 새로 학습합니다.")
        return None, None

def evaluate_model(model, X_test, y_test, test_df, feature_cols):
    """모델 성능 평가 (정확도만)"""
    
    if len(X_test) == 0:
        print("❌ 테스트 데이터가 없습니다!")
        return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    # 예측 수행
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).cpu().numpy().flatten()
    
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 정확도만 계산
    accuracy = accuracy_score(y_test, y_pred)
    
    # 결과 출력
    print("\n" + "="*50)
    print("📊 모델 성능 평가 결과")
    print("="*50)
    print(f"정확도 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 클래스별 성능
    print(f"\n클래스별 통계:")
    print(f"실제 라벨 분포: 하락={np.sum(y_test==0)}, 상승={np.sum(y_test==1)}")
    print(f"예측 라벨 분포: 하락={np.sum(y_pred==0)}, 상승={np.sum(y_pred==1)}")
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def simple_backtest(results, test_df, window=30):
    """간단한 백테스트 (수익률 계산)"""
    
    if len(test_df) <= window:
        print("❌ 백테스트를 위한 데이터가 충분하지 않습니다.")
        return None
    
    # 윈도우 크기만큼 조정된 테스트 데이터
    test_subset = test_df.iloc[window:].copy().reset_index(drop=True)
    
    if len(test_subset) != len(results['predictions']):
        print(f"⚠️ 데이터 길이 불일치: test_subset={len(test_subset)}, predictions={len(results['predictions'])}")
        return None
    
    # 예측 결과 추가
    test_subset['prediction'] = results['predictions']
    test_subset['probability'] = results['probabilities']
    
    # 실제 5일 후 수익률 계산
    test_subset = test_subset.sort_values(['ticker', 'date']).reset_index(drop=True)
    test_subset['actual_return'] = test_subset.groupby('ticker')['close'].transform(
        lambda x: (x.shift(-5) / x) - 1
    )
    
    # 유효한 데이터만 사용
    valid_data = test_subset.dropna(subset=['actual_return'])
    
    if len(valid_data) == 0:
        print("❌ 백테스트를 위한 유효한 데이터가 없습니다.")
        return None
    
    # 예측에 따른 수익률 계산
    buy_signals = valid_data[valid_data['prediction'] == 1]
    sell_signals = valid_data[valid_data['prediction'] == 0]
    
    if len(buy_signals) == 0:
        print("❌ 매수 신호가 없습니다.")
        return None
    
    # 성과 계산
    avg_return_buy = buy_signals['actual_return'].mean()
    avg_return_sell = sell_signals['actual_return'].mean() if len(sell_signals) > 0 else 0
    avg_return_total = valid_data['actual_return'].mean()
    
    # 승률 계산
    buy_win_rate = (buy_signals['actual_return'] > 0).mean()
    total_win_rate = (valid_data['actual_return'] > 0).mean()
    
    print("\n" + "="*50)
    print("📈 백테스트 결과")
    print("="*50)
    print(f"매수 신호 개수: {len(buy_signals)}")
    print(f"매도/관망 신호 개수: {len(sell_signals)}")
    print(f"매수 신호 평균 수익률: {avg_return_buy:.4f} ({avg_return_buy*100:.2f}%)")
    print(f"매수 신호 승률: {buy_win_rate:.4f} ({buy_win_rate*100:.2f}%)")
    print(f"전체 평균 수익률: {avg_return_total:.4f} ({avg_return_total*100:.2f}%)")
    print(f"전체 승률: {total_win_rate:.4f} ({total_win_rate*100:.2f}%)")
    print(f"전략 초과 수익률: {avg_return_buy - avg_return_total:.4f} ({(avg_return_buy - avg_return_total)*100:.2f}%)")
    
    return {
        'buy_signals': len(buy_signals),
        'sell_signals': len(sell_signals),
        'avg_return_buy': avg_return_buy,
        'avg_return_total': avg_return_total,
        'buy_win_rate': buy_win_rate,
        'total_win_rate': total_win_rate,
        'excess_return': avg_return_buy - avg_return_total
    }
    
def main():
    """메인 실행 함수"""
    
    print("데이터 로드 및 전처리 시작...")
    df = load_and_process_data()
    
    print("학습 데이터 준비...")
    result = prepare_training_data(df)
    
    if result[0] is None:
        print("❌ 데이터 준비 실패!")
        return None, None, None
    
    train_df, test_df, feature_cols, scaler = result
    
    # 기존 모델 확인
    model_path, scaler_path = check_model_exists()
    
    if model_path and scaler_path:
        # 기존 모델 로드
        model, scaler, feature_cols = load_trained_model(model_path, scaler_path)
        
        if model is None:
            print("❌ 모델 로드 실패! 새로 학습합니다.")
            # 새로 학습
            print("시계열 데이터 생성...")
            X_train, y_train = create_sequence(train_df, feature_cols, window=30)
            
            if len(X_train) == 0:
                print("❌ 시계열 데이터 생성 실패!")
                return None, None, None
            
            print(f"학습 데이터 형태: X_train {X_train.shape}, y_train {y_train.shape}")
            print(f"피처 개수: {len(feature_cols)}")
            
            print("모델 학습 시작...")
            model = train_model(X_train, y_train, epochs=50)
            
            print("모델 저장...")
            save_model_and_scaler(model, scaler, feature_cols)
        else:
            print("🔄 기존 모델을 사용하여 테스트합니다.")
    else:
        # 새로 학습
        print("시계열 데이터 생성...")
        X_train, y_train = create_sequence(train_df, feature_cols, window=30)
        
        if len(X_train) == 0:
            print("❌ 시계열 데이터 생성 실패!")
            return None, None, None
        
        print(f"학습 데이터 형태: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"피처 개수: {len(feature_cols)}")
        
        print("모델 학습 시작...")
        model = train_model(X_train, y_train, epochs=50)
        
        print("모델 저장...")
        save_model_and_scaler(model, scaler, feature_cols)
    
    # 테스트 데이터로 성능 평가
    if len(test_df) > 30:  # 윈도우 크기보다 큰 경우에만 평가
        print("\n테스트 데이터로 성능 평가...")
        X_test, y_test = create_sequence(test_df, feature_cols, window=30)
        
        if len(X_test) > 0:
            # 정확도 평가
            results = evaluate_model(model, X_test, y_test, test_df, feature_cols)
            
            if results:
                # 백테스트
                backtest_results = simple_backtest(results, test_df, window=30)
            
        else:
            print("❌ 테스트 데이터가 충분하지 않습니다.")
    else:
        print("❌ 테스트 데이터가 부족하여 성능 평가를 건너뜁니다.")
    
    print("✅ 완료!")
    
    return model, scaler, feature_cols

if __name__ == "__main__":
    model, scaler, feature_cols = main()