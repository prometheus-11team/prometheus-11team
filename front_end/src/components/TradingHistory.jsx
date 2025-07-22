import React from "react";
import "./TradingHistory.css";
import Header from "./Header";

const TradingHistory = () => {
  // Replace transactions with tradingHistory from Overview
  const tradingHistory = [
    {
      asset: "BTC/USD",
      type: "Buy",
      quantity: "0.5",
      price: "$30,000",
      datetime: "2024-01-15 10:00",
      profitLoss: "+$500",
    },
    {
      asset: "ETH/USD",
      type: "Sell",
      quantity: "2",
      price: "$1,800",
      datetime: "2024-01-16 14:30",
      profitLoss: "-$200",
    },
    {
      asset: "BTC/USD",
      type: "Sell",
      quantity: "0.2",
      price: "$31,000",
      datetime: "2024-01-17 09:15",
      profitLoss: "+$300",
    },
    {
      asset: "LTC/USD",
      type: "Buy",
      quantity: "5",
      price: "$100",
      datetime: "2024-01-18 11:45",
      profitLoss: "-$100",
    },
    {
      asset: "XRP/USD",
      type: "Buy",
      quantity: "100",
      price: "$0.50",
      datetime: "2024-01-19 16:00",
      profitLoss: "+$50",
    },
    {
      asset: "ETH/USD",
      type: "Buy",
      quantity: "1",
      price: "$1,900",
      datetime: "2024-01-20 12:20",
      profitLoss: "-$150",
    },
    {
      asset: "BTC/USD",
      type: "Buy",
      quantity: "0.3",
      price: "$32,000",
      datetime: "2024-01-21 08:50",
      profitLoss: "+$400",
    },
    {
      asset: "ADA/USD",
      type: "Sell",
      quantity: "50",
      price: "$0.40",
      datetime: "2024-01-22 15:10",
      profitLoss: "-$75",
    },
    {
      asset: "XRP/USD",
      type: "Sell",
      quantity: "200",
      price: "$0.55",
      datetime: "2024-01-23 10:30",
      profitLoss: "+$100",
    },
    {
      asset: "LTC/USD",
      type: "Sell",
      quantity: "10",
      price: "$110",
      datetime: "2024-01-24 13:55",
      profitLoss: "-$50",
    },
  ];

  // Original sample transactions (for below)
  const transactions = [
    {
      date: "2025-07-03",
      symbol: "AAPL",
      type: "매수",
      quantity: "4주",
      price: "190.1",
      profitLoss: null,
    },
    {
      date: "2025-07-03",
      symbol: "AAPL",
      type: "매수",
      quantity: "4주",
      price: "190.1",
      profitLoss: null,
    },
    {
      date: "2025-07-03",
      symbol: "AAPL",
      type: "매수",
      quantity: "4주",
      price: "190.1",
      profitLoss: null,
    },
    {
      date: "2025-07-03",
      symbol: "AAPL",
      type: "매수",
      quantity: "4주",
      price: "190.1",
      profitLoss: null,
    },
    {
      date: "2025-07-03",
      symbol: "AAPL",
      type: "매도",
      quantity: "4주",
      price: "190.1",
      profitLoss: "+20$",
    },
  ];

  return (
    <div className="trading-history-bg">
      <Header />
      <main className="trading-history-main">
        <div className="trading-history-container">
          <div className="trading-history-title">
            <span>Trading History</span>
          </div>
          <div className="trading-history-table">
            <div className="table-header">
              <div className="header-cell">Date/Time</div>
              <div className="header-cell">Asset</div>
              <div className="header-cell">Type</div>
              <div className="header-cell">Quantity</div>
              <div className="header-cell">Price</div>
              <div className="header-cell">Profit/Loss</div>
            </div>
            <div className="table-body">
              {tradingHistory.map((trade, index) => (
                <div key={index} className="table-row">
                  <div className="table-cell datetime-cell">{trade.datetime}</div>
                  <div className="table-cell asset-cell">{trade.asset}</div>
                  <div className="table-cell type-cell">{trade.type}</div>
                  <div className="table-cell quantity-cell">{trade.quantity}</div>
                  <div className="table-cell price-cell">{trade.price}</div>
                  <div className="table-cell profit-cell">{trade.profitLoss}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default TradingHistory;
