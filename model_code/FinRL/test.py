from stable_baselines3 import TD3
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.agents.stablebaselines3.models import DRLAgent
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# 저장된 모델 경로
model_save_path = os.path.join("models", "td3_model_with_macro")

# 주식 데이터 불러오기
df_stock = pd.read_csv("/Users/gamjawon/FinRL-Library/data/M7_stock_data_with_indicators.csv")
df_stock["date"] = pd.to_datetime(df_stock["date"])
df_stock = df_stock.rename(columns={"ticker": "tic"})
df_stock = df_stock.sort_values(["date", "tic"]).reset_index(drop=True)

# 거시경제 지표 불러오기
df_macro = pd.read_csv("/Users/gamjawon/FinRL-Library/data/macro_indicators_2020_2024.csv")
df_macro["date"] = pd.to_datetime(df_macro["date"])
df_macro = df_macro.sort_values("date").reset_index(drop=True)

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

df = merge_macro_data(df_stock, df_macro)
train_df = data_split(df, start="2020-01-01", end="2023-04-30")
test_df = data_split(df, start="2023-05-01", end="2025-04-30") 

basic_cols = ["date", "tic", "open", "high", "low", "close", "volume"]
# 기술지표 추출
TECH_INDICATORS = [col for col in df.columns if col not in basic_cols and col not in df_macro.columns]

# 거시경제 지표 추출 (date 제외)
MACRO_INDICATORS = [col for col in df_macro.columns if col != 'date']

# 모든 지표 통합
ALL_INDICATORS = TECH_INDICATORS + MACRO_INDICATORS

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

# 테스트 환경 다시 정의 (테스트셋 다시 불러야 할 수도 있음)
env_test = StockTradingEnv(df=test_df, **env_kwargs)

# 모델 불러오기
loaded_model = TD3.load(model_save_path)

# 예측 수행
print("\n📡 저장된 모델로 예측 시작...")
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=loaded_model, 
    environment=env_test
)
print("✅ 예측 완료!")

# ==========================
# 성과 지표 계산 함수들
# ==========================
def calculate_total_return(df):
    return (df["account_value"].iloc[-1] - df["account_value"].iloc[0]) / df["account_value"].iloc[0]

def calculate_cagr(df):
    start = df["account_value"].iloc[0]
    end = df["account_value"].iloc[-1]
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    return (end / start) ** (365.25 / days) - 1

def calculate_mdd(df):
    peak = df["account_value"].cummax()
    dd = df["account_value"] / peak - 1
    return dd.min()

def calculate_sharpe_ratio(df, risk_free_rate=0.02):
    returns = df["account_value"].pct_change().dropna()
    excess = returns - risk_free_rate / 252
    return excess.mean() / excess.std() * np.sqrt(252) if excess.std() != 0 else 0

def calculate_volatility(df):
    return df["account_value"].pct_change().dropna().std() * np.sqrt(252)

def calculate_benchmark_return(df_test, benchmark_tickers=["AAPL", "GOOGL", "MSFT"]):
    benchmark_returns = []
    for ticker in benchmark_tickers:
        df_tic = df_test[df_test["tic"] == ticker]
        if not df_tic.empty:
            start = df_tic.iloc[0]["close"]
            end = df_tic.iloc[-1]["close"]
            benchmark_returns.append((end - start) / start)
    avg_return = np.mean(benchmark_returns) if benchmark_returns else 0
    return avg_return, benchmark_returns

# ==========================
# 성과 분석 및 출력
# ==========================
total_return = calculate_total_return(df_account_value)
cagr = calculate_cagr(df_account_value)
mdd = calculate_mdd(df_account_value)
sharpe = calculate_sharpe_ratio(df_account_value)
volatility = calculate_volatility(df_account_value)
benchmark_return, individual_returns = calculate_benchmark_return(test_df)

initial_value = df_account_value["account_value"].iloc[0]
final_value = df_account_value["account_value"].iloc[-1]
profit = final_value - initial_value
outperformance = total_return - benchmark_return

print("=" * 60)
print("📊 성과 분석 결과")
print("=" * 60)
print(f"🤖 td3 모델 성과:")
print(f"   💰 총 수익률: {total_return:.2%}")
print(f"   📈 연복리 수익률 (CAGR): {cagr:.2%}")
print(f"   📉 최대 낙폭 (MDD): {mdd:.2%}")
print(f"   ⚡ 샤프 비율: {sharpe:.2f}")
print(f"   📊 변동성: {volatility:.2%}\n")

print("📊 벤치마크 성과:")
print(f"   🏆 평균 Buy & Hold 수익률: {benchmark_return:.2%}")
tickers = ["AAPL", "GOOGL", "MSFT"]
for i, r in enumerate(individual_returns):
    print(f"   📈 {tickers[i]} Buy & Hold: {r:.2%}")

print("\n🎯 상대 성과:")
print(f"   🚀 초과 수익률: {outperformance:.2%}")
print(f"   📈 성과: {'승리! 🎉' if outperformance > 0 else '아쉽... 📉'}")

print(f"\n💎 최종 결과:")
print(f"   🏦 초기 자금: {initial_value:,.0f}원")
print(f"   💰 최종 자금: {final_value:,.0f}원")
print(f"   💵 절대 수익: {profit:,.0f}원")
print("=" * 60)