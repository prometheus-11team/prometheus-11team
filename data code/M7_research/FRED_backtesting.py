import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 주가 데이터 로드
price_path = 'store data/M7_close_prices.csv'
prices = pd.read_csv(price_path, parse_dates=['Date'])

# 종목 리스트 추출
tickers = list(prices.columns)
tickers.remove('Date')

# 2. 매크로 지표 로드
macro_path = 'store data/macro_indicators_2020_2024.csv'
macro = pd.read_csv(macro_path, parse_dates=['Date'])

# 3. 시그널 생성 함수
def generate_improved_signals(macro_df):
    # 새로운 파생 지표 계산
    macro_df['Yield_Curve'] = macro_df['10Y Treasury Yield'] - macro_df['Federal Funds Rate']
    macro_df['CPI_MoM'] = macro_df['CPI'].pct_change()
    macro_df['Retail_Sales_MoM'] = macro_df['Retail Sales'].pct_change()
    macro_df['Retail_Sales_MA3'] = macro_df['Retail Sales'].rolling(window=3).mean()
    macro_df['M2_YoY'] = macro_df['M2 Money Stock'].pct_change(periods=12) * 100
    
    signals = []
    
    for i in range(len(macro_df)):
        score = 0
        
        # 기준금리 변화 및 수익률 곡선
        if i > 0 and macro_df.loc[i, 'Federal Funds Rate'] < macro_df.loc[i - 1, 'Federal Funds Rate']:
            score += 1
        elif i > 0 and macro_df.loc[i, 'Federal Funds Rate'] > macro_df.loc[i - 1, 'Federal Funds Rate']:
            score -= 1
            
        # 수익률 곡선 (신규)
        if not pd.isna(macro_df.loc[i, 'Yield_Curve']):
            if macro_df.loc[i, 'Yield_Curve'] > 1.0:  # 건강한 수익률 곡선
                score += 1
            elif macro_df.loc[i, 'Yield_Curve'] < 0:  # 역전된 수익률 곡선 (경기침체 신호)
                score -= 2  # 더 강한 부정적 신호
        
        # CPI - 절대값과 추세 모두 고려
        if macro_df.loc[i, 'CPI_MoM'] < 0:  # 디플레이션은 부정적
            score -= 1
        elif not pd.isna(macro_df.loc[i, 'CPI_MoM']) and macro_df.loc[i, 'CPI_MoM'] > 0 and macro_df.loc[i, 'CPI_MoM'] < 0.005:  # 적정 인플레이션
            score += 1
        elif not pd.isna(macro_df.loc[i, 'CPI_MoM']) and macro_df.loc[i, 'CPI_MoM'] >= 0.005:  # 높은 인플레이션
            score -= 1
            
        # 소매 판매 - 추세 중요
        if i >= 2:
            if not pd.isna(macro_df.loc[i, 'Retail_Sales_MoM']) and macro_df.loc[i, 'Retail_Sales_MoM'] > 0:
                score += 1
            elif not pd.isna(macro_df.loc[i, 'Retail_Sales_MoM']) and macro_df.loc[i, 'Retail_Sales_MoM'] < -0.01:  # 1% 이상 하락은 경고 신호
                score -= 1
        
        # 실업률 - 추세와 절대값 모두 중요
        if macro_df.loc[i, 'Unemployment Rate'] < 4.0:  # 매우 낮은 실업률
            score += 1
        elif macro_df.loc[i, 'Unemployment Rate'] > 5.0:  # 높은 실업률
            score -= 1
        
        if i > 0 and macro_df.loc[i, 'Unemployment Rate'] > macro_df.loc[i-1, 'Unemployment Rate'] + 0.3:  # 급격한 실업률 상승
            score -= 2  # 강한 부정적 신호
            
        # PCE 물가지수 - 추세 고려
        if i > 0 and not pd.isna(macro_df.loc[i, 'PCE Price Index']) and not pd.isna(macro_df.loc[i-1, 'PCE Price Index']):
            pce_mom = (macro_df.loc[i, 'PCE Price Index'] / macro_df.loc[i-1, 'PCE Price Index'] - 1) * 100
            if pce_mom > 0.5:  # 월 0.5% 이상 상승은 높은 인플레이션
                score -= 1
        
        # M2 통화량 - 적정 성장률 고려
        if not pd.isna(macro_df.loc[i, 'M2_YoY']):
            if macro_df.loc[i, 'M2_YoY'] > 0 and macro_df.loc[i, 'M2_YoY'] < 10:  # 적정 통화 성장
                score += 1
            elif macro_df.loc[i, 'M2_YoY'] > 15:  # 과도한 통화 공급
                score -= 1
            elif macro_df.loc[i, 'M2_YoY'] < -2:  # 통화량 급감
                score -= 2
        
        # 비농업 고용 - 추세 중요
        if i > 0 and macro_df.loc[i, 'Non-Farm Payrolls'] < macro_df.loc[i - 1, 'Non-Farm Payrolls']:
            score -= 1
        elif i > 0 and macro_df.loc[i, 'Non-Farm Payrolls'] > macro_df.loc[i - 1, 'Non-Farm Payrolls'] + 200:  # 강한 고용 증가
            score += 1
            
        # 신호 조건 재조정 (매수/매도 균형)
        if score >= 3:
            signals.append('BUY')
        elif score <= -3:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    macro_df['Signal'] = signals
    macro_df['Score'] = score  # 점수도 저장해 두면 분석에 유용
    return macro_df

# 4. 시그널 생성
macro = generate_improved_signals(macro)

# 5. 매크로 시그널과 주가 데이터 결합
data = pd.merge(prices, macro[['Date', 'Signal']], on='Date', how='inner')

# 6. 백테스트 함수
def backtest(stock_col, df, min_hold_days=5):
    position = 0
    cash = 1.0
    holdings = 0
    returns = []
    hold_days = 0

    for i in range(len(df)):
        price = df.iloc[i][stock_col]
        signal = df.iloc[i]['Signal']

        if signal == 'BUY' and position == 0:
            holdings = cash / price
            cash = 0
            position = 1
            hold_days = 0

        elif signal == 'SELL' and position == 1 and hold_days >= min_hold_days:
            cash = holdings * price
            holdings = 0
            position = 0

        if position == 1:
            hold_days += 1

        total_value = cash + (holdings * price if position == 1 else 0)
        returns.append(total_value)

    df['Portfolio Value'] = returns
    return df

# 7. 백테스트 실행 및 결과 시각화
results = {}
for stock in tickers:
    df_result = backtest(stock, data.copy())
    results[stock] = df_result

    plt.figure(figsize=(8, 4))
    plt.plot(df_result['Date'], df_result['Portfolio Value'], label=f'{stock} Portfolio')
    plt.title(f'Backtest Result for {stock}')
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

# 8. 최종 수익률 출력
for stock, df in results.items():
    print(f"{stock} Final Return: {df['Portfolio Value'].iloc[-1]:.2f}x")
    print(macro['Signal'].value_counts())

