import pandas as pd
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 디렉토리 생성
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =============================================================================
# 1. 데이터 로드 및 전처리
# =============================================================================

# 주식 데이터 불러오기
df_stock = pd.read_csv("/Users/gamjawon/prometheus-11team/FinRL-Library/examples/data/M7_stock_data_with_indicators.csv")
df_stock["date"] = pd.to_datetime(df_stock["date"])
df_stock = df_stock.rename(columns={"ticker": "tic"})
df_stock = df_stock.sort_values(["date", "tic"]).reset_index(drop=True)

# 거시경제 지표 불러오기
df_macro = pd.read_csv("/Users/gamjawon/prometheus-11team/DATA/technical/macro_indicators_2020_2025-03.csv")
df_macro["date"] = pd.to_datetime(df_macro["date"])
df_macro = df_macro.sort_values("date").reset_index(drop=True)

print("📊 거시경제 지표 데이터 확인:")
print(f"거시경제 지표 컬럼: {df_macro.columns.tolist()}")
print(f"거시경제 데이터 기간: {df_macro['date'].min()} ~ {df_macro['date'].max()}")
print(f"거시경제 데이터 샘플:\n{df_macro.head()}")

# =============================================================================
# 2. 거시경제 지표와 주식 데이터 병합
# =============================================================================

def merge_macro_data(df_stock, df_macro):
    """
    주식 데이터에 거시경제 지표를 병합하는 함수
    """
    # 거시경제 지표는 보통 월별/분기별 데이터이므로 forward fill 방식으로 병합
    df_merged = pd.merge_asof(
        df_stock.sort_values('date'), 
        df_macro.sort_values('date'),
        on='date',
        direction='backward'  # 가장 최근 발표된 거시지표 사용
    )
    
    # 거시경제 지표 결측치 처리
    macro_cols = [col for col in df_macro.columns if col != 'date']
    df_merged[macro_cols] = df_merged[macro_cols].fillna(method='ffill')
    df_merged[macro_cols] = df_merged[macro_cols].fillna(method='bfill')
    
    return df_merged

# 데이터 병합
df = merge_macro_data(df_stock, df_macro)

print(f"\n✅ 병합 완료! 총 컬럼 수: {len(df.columns)}")
print(f"병합된 데이터 기간: {df['date'].min()} ~ {df['date'].max()}")

# =============================================================================
# 3. 기술지표 및 거시지표 정의
# =============================================================================

# 기본 주가 데이터 컬럼
basic_cols = ["date", "tic", "open", "high", "low", "close", "volume"]

# 기술지표 추출
TECH_INDICATORS = [col for col in df.columns if col not in basic_cols and col not in df_macro.columns]

# 거시경제 지표 추출 (date 제외)
MACRO_INDICATORS = [col for col in df_macro.columns if col != 'date']

# 모든 지표 통합
ALL_INDICATORS = TECH_INDICATORS + MACRO_INDICATORS

print(f"\n📈 기술지표 ({len(TECH_INDICATORS)}개): {TECH_INDICATORS}")
print(f"🌍 거시지표 ({len(MACRO_INDICATORS)}개): {MACRO_INDICATORS}")

# =============================================================================
# 4. 데이터 분할
# =============================================================================

train_df = data_split(df, start="2020-01-01", end="2023-12-31")
test_df = data_split(df, start="2024-01-01", end="2025-04-30") 

print(f"\n📊 데이터 분할:")
print(f"훈련 데이터: {train_df['date'].min()} ~ {train_df['date'].max()} ({len(train_df)} rows)")
print(f"테스트 데이터: {test_df['date'].min()} ~ {test_df['date'].max()} ({len(test_df)} rows)")

# =============================================================================
# 5. 데이터 전처리
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
# 6. 환경 설정 (개선된 하이퍼파라미터)
# =============================================================================

stock_dim = len(train_df["tic"].unique())
state_space = 1 + 2 * stock_dim + len(ALL_INDICATORS) * stock_dim

env_kwargs = {
    "hmax": 100,  # 최대 보유량 증가
    "initial_amount": 1_000_000,
    "buy_cost_pct": [0.001] * stock_dim,
    "sell_cost_pct": [0.001] * stock_dim,
    "state_space": state_space,
    "stock_dim": stock_dim,
    "tech_indicator_list": ALL_INDICATORS,  # 모든 지표 사용
    "action_space": stock_dim,
    "reward_scaling": 1e-2,  # 보상 스케일링 증가
    "num_stock_shares": [0] * stock_dim,
    "turbulence_threshold": None,  
    "day": 0
}

print(f"\n🎯 환경 설정:")
print(f"주식 종목 수: {stock_dim}")
print(f"상태 공간 크기: {state_space}")
print(f"전체 지표 수: {len(ALL_INDICATORS)}")

# =============================================================================
# 7. 모델 학습
# =============================================================================

print(f"\n🚀 모델 학습 시작...")

env_train = StockTradingEnv(df=train_df, **env_kwargs)
agent = DRLAgent(env=env_train)

# PPO 모델 생성 및 학습
model = agent.get_model("td3")
trained_model = agent.train_model(
    model=model, 
    tb_log_name="td3_with_macro", 
    total_timesteps=100000  
)

print("✅ 학습 완료!")

# 모델 저장
model_save_path = os.path.join("models", "td3_model_with_macro")
trained_model.save(model_save_path)
print(f"\n💾 학습된 모델이 '{model_save_path}'에 저장되었습니다!")


# =============================================================================
# 8. 백테스팅
# =============================================================================

print(f"\n📊 백테스팅 시작...")

env_test = StockTradingEnv(df=test_df, **env_kwargs)
df_account_value, df_actions = agent.DRL_prediction(model=trained_model, environment=env_test)

print("✅ 백테스팅 완료!")

# =============================================================================
# 9. 성능 지표 계산 함수들
# =============================================================================

def calculate_total_return(df):
    """총 수익률 계산"""
    start_value = df["account_value"].iloc[0]
    end_value = df["account_value"].iloc[-1]
    total_return = (end_value - start_value) / start_value
    return total_return

def calculate_cagr(df):
    """연복리 수익률 계산"""
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
    """최대 낙폭 계산"""
    cumulative = df["account_value"].cummax()
    drawdown = df["account_value"] / cumulative - 1
    mdd = drawdown.min()
    return mdd

def calculate_sharpe_ratio(df, risk_free_rate=0.02):
    """샤프 비율 계산"""
    returns = df["account_value"].pct_change().dropna()
    excess_returns = returns - risk_free_rate/252  # 일간 무위험 수익률
    if excess_returns.std() != 0:
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    return sharpe

def calculate_volatility(df):
    """변동성 계산"""
    returns = df["account_value"].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    return volatility

def calculate_benchmark_return(df_test, benchmark_tickers=["AAPL", "GOOGL", "MSFT"]):
    """벤치마크 수익률 계산 (복수 종목 평균)"""
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

# =============================================================================
# 10. 성과 분석
# =============================================================================

print(f"\n" + "="*60)
print(f"📊 성과 분석 결과")
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
print(f"🤖 td3 모델 성과:")
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
# 11. 시각화
# =============================================================================

plt.figure(figsize=(15, 10))

# 1. 포트폴리오 가치 변화
plt.subplot(2, 2, 1)
plt.plot(df_account_value["date"], df_account_value["account_value"], 'b-', linewidth=2, label='DRL Portfolio')
plt.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.7, label='Initial Value')
plt.title('📈 Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (원)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 2. 수익률 변화
plt.subplot(2, 2, 2)
returns = df_account_value["account_value"].pct_change().dropna()
cumulative_returns = (1 + returns).cumprod() - 1
plt.plot(df_account_value["date"][1:], cumulative_returns * 100, 'g-', linewidth=2)
plt.title('📊 Cumulative Returns (%)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 3. 드로우다운
plt.subplot(2, 2, 3)
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
plt.subplot(2, 2, 4)
if len(df_actions) > 0:
    # 각 종목별 거래량 합계
    action_cols = [col for col in df_actions.columns if col != 'date']
    daily_trades = df_actions[action_cols].abs().sum(axis=1)
    plt.plot(range(len(daily_trades)), daily_trades, 'purple', linewidth=1)
    plt.title('🔄 Daily Trading Activity')
    plt.xlabel('Time steps')
    plt.ylabel('Total Trades')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("results/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\n💾 결과가 'results/comprehensive_analysis.png'로 저장되었습니다!")
print(f"="*60)