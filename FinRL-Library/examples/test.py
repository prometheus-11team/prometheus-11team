import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from datetime import timedelta
from pipline import (
    HierarchicalTradingSystem,
    load_trained_model,
    evaluate_model_performance,
    load_market_data
)

# ===============================
# ✅ 경로 설정
# ===============================
transformer_model_path = "/Users/gamjawon/FinRL-Library/examples/models/transformer_classifier.pt"
trained_model_path = "trained_hierarchical_model.pth"
market_data_path = "/Users/gamjawon/FinRL-Library/examples/data/merged.csv"
sentiment_data_paths = [
    "/Users/gamjawon/FinRL-Library/examples/data/M7_googlenews_2020_2022_sentiment_feature.csv",
    "/Users/gamjawon/FinRL-Library/examples/data/M7_reddits_2022_2025_sentiment_feature.csv"
]

# ===============================
# ✅ 테스트 데이터 로딩 및 분리
# ===============================
market_data_all = load_market_data(market_data_path)
market_data_all['date'] = pd.to_datetime(market_data_all['date'])
test_data = market_data_all[market_data_all['date'] >= '2023-07-01'].copy()
print(f"📆 테스트 데이터 범위: {test_data['date'].min().date()} ~ {test_data['date'].max().date()} ({len(test_data)} rows)")

# ===============================
# ✅ TransformerClassifier 정의
# ===============================
class TransformerClassifier(nn.Module):
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

        self.predict_proba = self._predict_proba

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embed(x)
        x = self.encoder(x)
        return torch.sigmoid(self.head(x[:, -1]))

    def _predict_proba(self, x):
        with torch.no_grad():
            self.eval()
            prob_positive = self.forward(x).cpu().numpy()
            prob_negative = 1 - prob_positive
            return np.column_stack([prob_negative, prob_positive])

# ===============================
# ✅ 저장된 모델 로드 (weights_only=False 설정 필요)
# ===============================
trading_system = load_trained_model(
    trained_model_path,
    transformer_model_path,
    market_data=test_data,
    sentiment_data_paths=sentiment_data_paths,
    weights_only=False  # 이 인자에 맞춰 pipline.py 함수도 수정되어 있어야 함
)

# ===============================
# ✅ 성능 평가
# ===============================
performance_result = evaluate_model_performance(trading_system, test_data, test_period_days=30)

# ===============================
# ✅ 평가 결과 출력
# ===============================
if performance_result:
    print("\n📊 최종 성능 요약:")
    print(f"  🔹 총 수익률: {performance_result['total_return']:.4f}")
    print(f"  🔹 평균 일일 수익률: {performance_result['avg_daily_return']:.4f}")
    print(f"  🔹 변동성: {performance_result['volatility']:.4f}")
    print(f"  🔹 샤프 비율: {performance_result['sharpe_ratio']:.4f}")
