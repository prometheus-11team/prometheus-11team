import React from "react";
import "./Overview.css";
import Header from "./Header";

const Overview = () => {
  const stockData = [
    {
      symbol: "AAPL",
      techScore: "0.82",
      sentimentScore: "0.7",
      profit: "+2.5%",
      profitClass: "profit-positive",
      quantity: "8주",
      signal: "매수",
      signalClass: "signal-buy",
    },
    {
      symbol: "MSFT",
      techScore: "0.75",
      sentimentScore: "0.5",
      profit: "+3.1%",
      profitClass: "profit-positive",
      quantity: "5주",
      signal: "Hold",
      signalClass: "signal-hold",
    },
    {
      symbol: "TSLA",
      techScore: "0.61",
      sentimentScore: "0.2",
      profit: "-1.8%",
      profitClass: "profit-negative",
      quantity: "3주",
      signal: "매도",
      signalClass: "signal-sell",
    },
    {
      symbol: "AMZN",
      techScore: "0.68",
      sentimentScore: "-0.3",
      profit: "+0.9%",
      profitClass: "profit-positive",
      quantity: "2주",
      signal: "Hold",
      signalClass: "signal-hold",
    },
    {
      symbol: "GOOGL",
      techScore: "0.59",
      sentimentScore: "0.1",
      profit: "-2.2%",
      profitClass: "profit-negative",
      quantity: "없음",
      signal: "매수",
      signalClass: "signal-buy",
    },
    {
      symbol: "NVDA",
      techScore: "0.91",
      sentimentScore: "0.8",
      profit: "+7.2%",
      profitClass: "profit-positive",
      quantity: "4주",
      signal: "매수",
      signalClass: "signal-buy",
    },
    {
      symbol: "META",
      techScore: "0.77",
      sentimentScore: "0.6",
      profit: "+4.2%",
      profitClass: "profit-positive",
      quantity: "5주",
      signal: "Hold",
      signalClass: "signal-hold",
    },
  ];

  return (
    <div className="overview-container">
      <div className="overview-background">
        <div className="overview-layout">
          <Header />

          {/* Main Content */}
          <main className="overview-main">
            <div className="overview-content">
              {/* Page Title */}
              <div className="page-title-section">
                <div className="title-content">
                  <div className="title-container">
                    <h2 className="page-title">Stocks Overview</h2>
                  </div>
                  <div className="subtitle-container">
                    <p className="page-subtitle">
                      View your past trades and performance.
                    </p>
                  </div>
                </div>
              </div>

              {/* Stock Cards Section */}
              <div className="stocks-section">
                <div className="stock-cards-container">
                  {/* 3-column grid, last row centered if not full */}
                  <div className="stock-cards-row">
                    {stockData.slice(0, 3).map((stock, index) => (
                      <div key={index} className="stock-card">
                        <div className="stock-symbol">{stock.symbol}</div>
                        <div className="stock-tech-score">
                          기술 점수: {stock.techScore}
                        </div>
                        <div className="stock-sentiment-score">
                          감성 점수: {stock.sentimentScore}
                        </div>
                        <div className={`stock-profit ${stock.profitClass}`}>
                          수익률: {stock.profit}
                        </div>
                        <div className="stock-quantity">
                          보유량: {stock.quantity}
                        </div>
                        <div className={`stock-signal ${stock.signalClass}`}>
                          신호: {stock.signal}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="stock-cards-row">
                    {stockData.slice(3, 6).map((stock, index) => (
                      <div key={index} className="stock-card">
                        <div className="stock-symbol">{stock.symbol}</div>
                        <div className="stock-tech-score">
                          기술 점수: {stock.techScore}
                        </div>
                        <div className="stock-sentiment-score">
                          감성 점수: {stock.sentimentScore}
                        </div>
                        <div className={`stock-profit ${stock.profitClass}`}>
                          수익률: {stock.profit}
                        </div>
                        <div className="stock-quantity">
                          보유량: {stock.quantity}
                        </div>
                        <div className={`stock-signal ${stock.signalClass}`}>
                          신호: {stock.signal}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="stock-cards-row single-card">
                    {stockData.slice(6, 7).map((stock, index) => (
                      <div key={index} className="stock-card">
                        <div className="stock-symbol">{stock.symbol}</div>
                        <div className="stock-tech-score">
                          기술 점수: {stock.techScore}
                        </div>
                        <div className="stock-sentiment-score">
                          감성 점수: {stock.sentimentScore}
                        </div>
                        <div className={`stock-profit ${stock.profitClass}`}>
                          수익률: {stock.profit}
                        </div>
                        <div className="stock-quantity">
                          보유량: {stock.quantity}
                        </div>
                        <div className={`stock-signal ${stock.signalClass}`}>
                          신호: {stock.signal}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
};

export default Overview;
