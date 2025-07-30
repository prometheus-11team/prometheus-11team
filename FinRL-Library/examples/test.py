import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from sklearn.preprocessing import StandardScaler
from finrl.meta.preprocessor.preprocessors import data_split
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 모델 경로
MODEL_PATH = os.path.join(BASE_DIR, "models/td3_enhanced_model(4)")
TRANSFORMER_PATH = os.path.join(BASE_DIR, "../../model/transformer_classifier_best.pt")

# 데이터 경로
stock_data_path = os.path.join(BASE_DIR, "../../FinRL-Library/examples/data/M7_stock_data_with_indicators.csv")
macro_data_path = os.path.join(BASE_DIR, "../../DATA/technical/macro_indicators_2020_2025-03.csv")
financial_data_path = os.path.join(BASE_DIR, "../../FinRL-Library/examples/data/M7_financial_data_2020_2025.csv")

sentiment_data_paths = [
    os.path.join(BASE_DIR, "../../FinRL-Library/examples/data/M7_googlenews_2020_2022_sentiment_feature.csv"),
    os.path.join(BASE_DIR, "../../FinRL-Library/examples/data/M7_reddits_2022_2025_sentiment_feature.csv")
]

# 필요한 클래스와 함수 (TransformerClassifier, EnhancedDataProcessor)는 기존 코드에서 import 하거나 복사 필요
from run_mine import TransformerClassifier, EnhancedDataProcessor  # 반드시 맞게 경로 수정

def calculate_total_return(df):
    return (df["account_value"].iloc[-1] - df["account_value"].iloc[0]) / df["account_value"].iloc[0]

def calculate_cagr(df):
    start_value = df["account_value"].iloc[0]
    end_value = df["account_value"].iloc[-1]
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    years = days / 365.25
    return (end_value / start_value) ** (1 / years) - 1 if years > 0 else 0

def calculate_mdd(df):
    peak = df["account_value"].cummax()
    dd = df["account_value"] / peak - 1
    return dd.min()

def calculate_sharpe_ratio(df, risk_free_rate=0.02):
    returns = df["account_value"].pct_change().dropna()
    excess = returns - risk_free_rate / 252
    return excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

def calculate_volatility(df):
    return df["account_value"].pct_change().dropna().std() * np.sqrt(252)

def calculate_benchmark_return(df_test, benchmark_tickers=["AAPL", "GOOGL", "MSFT"]):
    returns = []
    for ticker in benchmark_tickers:
        df_t = df_test[df_test["tic"] == ticker]
        if len(df_t) > 0:
            buy = df_t.iloc[0]["close"]
            sell = df_t.iloc[-1]["close"]
            returns.append((sell - buy) / buy)
    return np.mean(returns), returns

def save_portfolio_value_plot(df_account_value, output_path="results/test_portfolio_value.png"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(df_account_value["date"], df_account_value["account_value"], label="Portfolio Value", color="blue", linewidth=2)
    plt.axhline(y=df_account_value["account_value"].iloc[0], color='gray', linestyle='--', label="Initial Value")
    plt.title("📈 Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Account Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"💾 포트폴리오 가치 그래프 저장 완료: {output_path}")

def main():
    print("📊 확장된 TD3 모델 백테스팅만 실행합니다...")

    # 데이터 로딩 및 병합
    processor = EnhancedDataProcessor(transformer_model_path=TRANSFORMER_PATH)
    df = processor.load_and_merge_data(stock_data_path, macro_data_path, financial_data_path, sentiment_data_paths)

    # 인디케이터 선택
    tech_indicators = ['SMA_50', 'SMA_200', 'RSI_14', 'ROC_10', 'MACD', 'MACD_Signal']
    tech_indicators = [col for col in tech_indicators if col in df.columns]
    macro_indicators = [
        'Federal Funds Rate', '10Y Treasury Yield', 'CPI', 'Core CPI', 
        'PCE Price Index', 'Retail Sales', 'Unemployment Rate', 
        'Non-Farm Payrolls', 'M2 Money Stock'
    ]
    macro_indicators = [col for col in macro_indicators if col in df.columns]
    transformer_indicators = ['transformer_prediction', 'transformer_confidence', 'transformer_signal']
    sentiment_indicators = [col for col in df.columns if any(x in col for x in ['positive', 'negative', 'neutral'])]
    ALL_INDICATORS = tech_indicators + macro_indicators + transformer_indicators + sentiment_indicators

    # 테스트 데이터 분리
    test_df = data_split(df, start="2024-01-01", end="2025-04-30")
    test_df = test_df.fillna(method="ffill").fillna(method="bfill").dropna(subset=ALL_INDICATORS)
    test_df.reset_index(drop=True, inplace=True)
    test_df.index = test_df["date"].factorize()[0]

    stock_dim = len(test_df["tic"].unique())
    state_space = 1 + 2 * stock_dim + len(ALL_INDICATORS) * stock_dim

    env_kwargs = {
        "hmax": 800,
        "initial_amount": 1_000_000,
        "buy_cost_pct": [0.0005] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": ALL_INDICATORS,
        "action_space": stock_dim,
        "reward_scaling": 2e-1,
        "num_stock_shares": [0] * stock_dim,
        "turbulence_threshold": None,
        "day": 0
    }

    env_test = StockTradingEnv(df=test_df, **env_kwargs)
    agent = DRLAgent(env=env_test)

    print("📦 모델 로드 중...")
    from stable_baselines3 import TD3
    model = TD3.load(MODEL_PATH)

    df_account_value, df_actions = agent.DRL_prediction(model=model, environment=env_test)
    print("✅ 백테스팅 완료!")

    # 지표 분석
    total_return = calculate_total_return(df_account_value)
    cagr = calculate_cagr(df_account_value)
    mdd = calculate_mdd(df_account_value)
    sharpe = calculate_sharpe_ratio(df_account_value)
    volatility = calculate_volatility(df_account_value)
    benchmark_return, individual_returns = calculate_benchmark_return(test_df)

    print("\n📊 백테스팅 성과:")
    print(f"   💰 총 수익률: {total_return:.2%}")
    print(f"   📈 CAGR: {cagr:.2%}")
    print(f"   📉 최대 낙폭: {mdd:.2%}")
    print(f"   ⚡ Sharpe Ratio: {sharpe:.2f}")
    print(f"   📊 Volatility: {volatility:.2%}")
    print(f"   🏆 Benchmark 평균 수익률: {benchmark_return:.2%}")

    outperformance = total_return - benchmark_return
    print(f"   🚀 초과 수익률: {outperformance:.2%}")

    # 파일 저장
    df_account_value.to_csv("results/test_account_value.csv", index=False)
    df_actions.to_csv("results/test_trading_actions.csv", index=False)
    save_portfolio_value_plot(df_account_value, "results/test_portfolio_value.png")
    print("💾 결과 저장 완료: test_account_value.csv, test_trading_actions.csv")
    

if __name__ == "__main__":
    main()
