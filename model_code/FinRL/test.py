import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from pipline import (
    HierarchicalTradingSystem, 
    load_market_data, 
    load_trained_model
)

class EnhancedTradingBacktester:
    """
    학습된 계층적 거래 시스템 백테스터 (RL + Transformer + 감정분석)
    """
    
    def __init__(self, trained_model_path, transformer_model_path, 
                 market_data, sentiment_data_paths):
        """
        Args:
            trained_model_path: 학습된 RL 모델 경로
            transformer_model_path: Transformer 모델 경로  
            market_data: 시장 데이터
            sentiment_data_paths: 감정 분석 데이터 경로들
        """
        print("🤖 학습된 모델 로드 중...")
        
        # 학습된 모델 로드
        self.trading_system = load_trained_model(
            trained_model_path,
            transformer_model_path, 
            market_data,
            sentiment_data_paths,
            weights_only=False
        )
        
        if self.trading_system is None:
            raise ValueError("❌ 학습된 모델 로드 실패")
            
        self.test_results = None
        self.signals_df = None
        
        print("✅ 학습된 RL 모델 로드 완료!")
        
    def prepare_test_data_with_rl(self, test_data):
        """RL 모델을 사용한 테스트 데이터 준비"""
        print("🧠 RL 모델로 거래 신호 생성 중...")
        
        signals = []
        probabilities = []
        decisions = []
        
        # 각 데이터 포인트에 대해 RL 결정 생성
        for idx, row in test_data.iterrows():
            try:
                date = row['Date']
                ticker = row['Ticker']
                
                # RL 모델의 거래 결정 획득
                decision = self.trading_system.make_trading_decision(date, ticker)
                
                # 결정을 숫자 신호로 변환
                if decision['action'] == 'BUY':
                    signal = 1
                    prob = 0.5 + decision['quantity'] * 0.5  # 0.5~1.0
                elif decision['action'] == 'SELL':
                    signal = 0  # 매도는 0으로 처리 (롱 온리 전략)
                    prob = 0.5 - decision['quantity'] * 0.5  # 0.0~0.5
                else:  # HOLD
                    signal = 0
                    prob = 0.5
                
                signals.append(signal)
                probabilities.append(prob)
                decisions.append(decision)
                
            except Exception as e:
                print(f"⚠️ 신호 생성 실패 ({date}, {ticker}): {e}")
                signals.append(0)
                probabilities.append(0.5)
                decisions.append({'action': 'HOLD', 'confidence': 'LOW', 'quantity': 0})
        
        # 테스트 데이터에 RL 예측 결과 추가
        test_df_with_pred = test_data.copy()
        test_df_with_pred['RL_Signal'] = signals
        test_df_with_pred['RL_Proba'] = probabilities
        test_df_with_pred['Signal'] = signals  # 백테스트용
        
        # 상세 신호 DataFrame 생성
        signals_df = pd.DataFrame({
            'Date': test_data['Date'],
            'Ticker': test_data['Ticker'],
            'Signal': signals,
            'Proba': probabilities,
            'Action': [d['action'] for d in decisions],
            'Confidence': [d['confidence'] for d in decisions],
            'Quantity': [d['quantity'] for d in decisions]
        })
        
        self.signals_df = signals_df
        
        print(f"✅ RL 신호 생성 완료:")
        print(f"   - BUY 신호: {sum(signals)} 개")
        print(f"   - 전체 신호: {len(signals)} 개")
        print(f"   - 매수 비율: {sum(signals)/len(signals)*100:.1f}%")
        
        return test_df_with_pred, signals_df
    
    def run_backtest(self, test_df_with_pred, holding_period=5):
        """백테스트 실행 (기존과 동일)"""
        print(f"🚀 RL 모델 백테스트 실행 중... (보유기간: {holding_period}일)")
        
        # 1. 진입가/청산가 계산
        test_df = (
            test_df_with_pred
            .sort_values(['Ticker', 'Date'])
            .groupby('Ticker')
            .apply(lambda d: d.assign(
                entry_price=d['Close'].shift(-1),  # 다음날 시가로 진입
                exit_price=d['Close'].shift(-holding_period)  # N일 후 종가로 청산
            ))
            .reset_index(drop=True)
        )
        
        # 2. 개별 트레이드 수익률 계산
        test_df['trade_return'] = np.where(
            test_df['Signal'] == 1,  # 매수 신호일 때만
            (test_df['exit_price'] / test_df['entry_price'] - 1),
            0.0  # 매수 신호가 아니면 수익률 0
        )
        
        # NaN 값 처리
        test_df['trade_return'] = test_df['trade_return'].fillna(0)
        
        # 3. 일별 포트폴리오 수익률 집계
        daily_returns = (
            test_df
            .dropna(subset=['entry_price', 'exit_price'])
            .groupby('Date')['trade_return']
            .mean()  # 일별 평균 수익률
            .fillna(0)
        )
        
        # 4. 성과지표 계산
        if len(daily_returns) == 0:
            print("❌ 백테스트 데이터가 없습니다.")
            return None
        
        # 누적수익률
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        
        # 기본 성과지표
        total_return = cumulative_returns.iloc[-1]
        trading_days = len(daily_returns)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1
        
        # 위험 조정 수익률
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return / volatility) if volatility > 0 else 0
        
        # 최대 낙폭 (Maximum Drawdown)
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / (1 + rolling_max)
        max_drawdown = drawdown.min()
        
        # 승률 계산
        profitable_trades = test_df[test_df['trade_return'] > 0]
        total_trades = test_df[test_df['Signal'] == 1]
        win_rate = len(profitable_trades) / len(total_trades) if len(total_trades) > 0 else 0
        
        # 결과 저장
        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(total_trades),
            'trading_days': trading_days,
            'daily_returns': daily_returns,
            'cumulative_returns': cumulative_returns,
            'detailed_trades': test_df,
            'drawdown_series': drawdown
        }
        
        self.test_results = results
        
        # 5. 결과 출력
        self._print_results(results)
        
        return results
    
    def _print_results(self, results):
        """백테스트 결과 출력 (기존과 동일)"""
        print("\n" + "="*60)
        print("🤖 RL 모델 백테스트 결과 요약")
        print("="*60)
        print(f"📊 총 수익률      : {results['total_return']:.2%}")
        print(f"📊 연환산 수익률   : {results['annual_return']:.2%}")
        print(f"📊 변동성         : {results['volatility']:.2%}")
        print(f"📊 샤프 비율      : {results['sharpe_ratio']:.3f}")
        print(f"📊 최대 낙폭      : {results['max_drawdown']:.2%}")
        print(f"📊 승률          : {results['win_rate']:.2%}")
        print(f"📊 총 거래 수     : {results['total_trades']:,}")
        print(f"📊 거래 기간      : {results['trading_days']} 거래일")
        print("="*60)
        
        # 월별 수익률
        if len(results['daily_returns']) > 0:
            monthly_returns = results['daily_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            print(f"\n📅 월별 수익률 (최근 12개월):")
            for date, ret in monthly_returns.tail(12).items():
                print(f"   {date.strftime('%Y-%m')}: {ret:.2%}")
    
    def analyze_rl_decisions(self):
        """RL 결정 분석"""
        if self.signals_df is None:
            print("❌ 신호 데이터가 없습니다.")
            return
        
        print("\n" + "="*50)
        print("🧠 RL 모델 결정 분석")
        print("="*50)
        
        # 행동별 분포
        action_counts = self.signals_df['Action'].value_counts()
        print("📊 행동별 분포:")
        for action, count in action_counts.items():
            percentage = count / len(self.signals_df) * 100
            print(f"   {action}: {count}건 ({percentage:.1f}%)")
        
        # 신뢰도별 분포
        confidence_counts = self.signals_df['Confidence'].value_counts()
        print("\n🎯 신뢰도별 분포:")
        for conf, count in confidence_counts.items():
            percentage = count / len(self.signals_df) * 100
            print(f"   {conf}: {count}건 ({percentage:.1f}%)")
        
        # 종목별 BUY 신호 빈도
        buy_signals = self.signals_df[self.signals_df['Signal'] == 1]
        if len(buy_signals) > 0:
            ticker_buy_counts = buy_signals['Ticker'].value_counts()
            print("\n📈 종목별 BUY 신호 빈도:")
            for ticker, count in ticker_buy_counts.items():
                print(f"   {ticker}: {count}건")
        
        # 수량 분포 히스토그램
        quantities = self.signals_df['Quantity']
        print(f"\n📏 거래 수량 통계:")
        print(f"   평균: {quantities.mean():.3f}")
        print(f"   표준편차: {quantities.std():.3f}")
        print(f"   최대: {quantities.max():.3f}")
        print(f"   최소: {quantities.min():.3f}")
    
    def plot_results(self, save_path=None):
        """백테스트 결과 시각화 (기존과 동일하되 제목 변경)"""
        if self.test_results is None:
            print("❌ 백테스트를 먼저 실행해주세요.")
            return
        
        results = self.test_results
        
        # 플롯 설정
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🤖 RL 모델 백테스트 결과 분석', fontsize=16, fontweight='bold')
        
        # 1. 누적수익률 곡선
        ax1 = axes[0, 0]
        cumulative_returns = results['cumulative_returns']
        ax1.plot(cumulative_returns.index, cumulative_returns * 100, linewidth=2, color='blue')
        ax1.set_title('누적 수익률', fontweight='bold')
        ax1.set_ylabel('수익률 (%)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 드로우다운 곡선
        ax2 = axes[0, 1]
        drawdown = results['drawdown_series']
        ax2.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown * 100, color='red', linewidth=1)
        ax2.set_title('드로우다운', fontweight='bold')
        ax2.set_ylabel('드로우다운 (%)')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 일별 수익률 히스토그램
        ax3 = axes[1, 0]
        daily_returns = results['daily_returns'] * 100
        ax3.hist(daily_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(daily_returns.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'평균: {daily_returns.mean():.2f}%')
        ax3.set_title('일별 수익률 분포', fontweight='bold')
        ax3.set_xlabel('일별 수익률 (%)')
        ax3.set_ylabel('빈도')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 월별 수익률 바차트
        ax4 = axes[1, 1]
        monthly_returns = results['daily_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        colors = ['green' if x > 0 else 'red' for x in monthly_returns]
        bars = ax4.bar(range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7)
        ax4.set_title('월별 수익률', fontweight='bold')
        ax4.set_xlabel('월')
        ax4.set_ylabel('수익률 (%)')
        ax4.grid(True, alpha=0.3)
        
        # x축 라벨을 월로 설정
        if len(monthly_returns) > 0:
            ax4.set_xticks(range(0, len(monthly_returns), max(1, len(monthly_returns)//12)))
            ax4.set_xticklabels([monthly_returns.index[i].strftime('%Y-%m') 
                                for i in range(0, len(monthly_returns), max(1, len(monthly_returns)//12))],
                               rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 차트 저장: {save_path}")
        
        plt.show()

def main_rl_backtest():
    """RL 모델 백테스트 메인 실행 함수"""
    
    # 파일 경로 설정
    trained_model_path = '/Users/gamjawon/prometheus-11team/FinRL-Library/examples/trained_hierarchical_model.pth'
    transformer_model_path = '/Users/gamjawon/prometheus-11team/model/transformer_classifier_best.pt'
    market_data_path = "/Users/gamjawon/prometheus-11team/DATA/technical/M7_stock_data_with_indicators.csv"
    sentiment_data_paths = [
        "/Users/gamjawon/prometheus-11team/DATA/NLP/M7_googlenews_2020_2022_sentiment_feature.csv",
        "/Users/gamjawon/prometheus-11team/DATA/NLP/M7_reddits_2022_2025_sentiment_feature.csv"
    ]
    
    print("🤖 RL 모델 백테스트 시작...")
    
    # 1. 데이터 로드
    market_data_raw = load_market_data(market_data_path)
    
    if market_data_raw.empty:
        print("❌ 시장 데이터 로드 실패")
        return
    
    # 2. 필요한 컬럼만 선택
    required_cols = ['Date', 'Ticker', 'Close', 'ROC_10', 'RSI_14', 'MACD']
    market_data_filtered = market_data_raw[required_cols].copy()
    
    # 3. RL 백테스터 초기화 (학습된 모델 로드)
    try:
        backtester = EnhancedTradingBacktester(
            trained_model_path,
            transformer_model_path,
            market_data_filtered,
            sentiment_data_paths
        )
    except Exception as e:
        print(f"❌ RL 백테스터 초기화 실패: {e}")
        return
    
    # 4. 테스트 데이터 분리
    split_date = '2024-01-01'
    full_data = backtester.trading_system.env.market_data
    test_data = full_data[full_data['Date'] >= split_date].copy().reset_index(drop=True)
    
    print(f"✅ 테스트 데이터: {len(test_data)} rows")
    
    # 5. RL 모델로 테스트 데이터 준비
    test_df_with_pred, signals_df = backtester.prepare_test_data_with_rl(test_data)
    
    if test_df_with_pred is not None:
        # 6. 백테스트 실행
        results = backtester.run_backtest(test_df_with_pred, holding_period=5)
        
        if results:
            # 7. 시각화
            backtester.plot_results(save_path='rl_backtest_results.png')
            
            # 8. RL 결정 분석
            backtester.analyze_rl_decisions()
            
            print("\n✅ RL 모델 백테스트 완료!")
            print(f"📁 결과 저장: rl_backtest_results.png")
        else:
            print("❌ 백테스트 실행 실패")
    else:
        print("❌ 테스트 데이터 준비 실패")

if __name__ == "__main__":
    main_rl_backtest()