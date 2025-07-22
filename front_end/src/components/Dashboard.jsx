import React from "react";
// import { Link } from "react-router-dom";
import "./Dashboard.css";
import Header from "./Header";

const Dashboard = () => {
  return (
    <div className="dashboard-container">
      <div className="dashboard-background">
        <Header />
        <main className="dashboard-main">
          <h2 className="dashboard-title">Dashboard</h2>
          <div className="metrics-section">
            <div className="metrics-card">
              <div className="card-label">Total Profit/Loss</div>
              <div className="card-value">$12,500</div>
              <div className="card-change positive">+15%</div>
            </div>
            <div className="metrics-card">
              <div className="card-label">Current Balance</div>
              <div className="card-value">$50,000</div>
              <div className="card-change positive">+10,000</div>
            </div>
            <div className="metrics-card">
              <div className="card-label">Sharp ratio</div>
              <div className="card-value">1.24</div>
              <div className="card-change positive">+0.2</div>
            </div>
          </div>
          <div className="chart-section">
            <div className="chart-header">
              <div>
                <div className="chart-title">Performance Chart</div>
                <div className="chart-percentage">+15%</div>
                <div className="chart-period">
                  Last 30 Days <span className="positive">+15%</span>
                </div>
              </div>
            </div>
            <div className="chart-visualization">
              {/* SVG 차트 그대로 사용 */}
              <svg
                className="performance-chart"
                width="852"
                height="186"
                viewBox="0 0 852 186"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <g clipPath="url(#clip0_15_687)">
                  <path
                    fillRule="evenodd"
                    clipRule="evenodd"
                    d="M5.1841 188.356C36.5545 188.356 36.5545 36.2887 67.925 36.2887C99.2954 36.2887 99.2954 70.8494 130.666 70.8494C162.036 70.8494 162.036 160.707 193.407 160.707C224.777 160.707 224.777 57.0251 256.148 57.0251C287.519 57.0251 287.519 174.531 318.888 174.531C350.259 174.531 350.259 105.41 381.629 105.41C413 105.41 413 77.7615 444.371 77.7615C475.741 77.7615 475.741 209.092 507.112 209.092C538.481 209.092 538.481 257.477 569.852 257.477C601.223 257.477 601.223 1.72803 632.593 1.72803C663.964 1.72803 663.964 139.971 695.335 139.971C726.704 139.971 726.704 222.916 758.074 222.916C789.445 222.916 789.445 43.2008 820.816 43.2008V257.477H569.852H5.1841V188.356Z"
                    fill="url(#paint0_linear_15_687)"
                  />
                  <path
                    d="M11.5 145.506C42.8704 145.506 36.5545 36.2887 67.925 36.2887C99.2954 36.2887 99.2954 70.8494 130.666 70.8494C162.036 70.8494 164.13 145.506 195.5 145.506C226.871 145.506 224.777 28 256.148 28C287.519 28 287.519 145.506 318.888 145.506C350.259 145.506 350.259 105.41 381.629 105.41C413 105.41 413 77.7615 444.371 77.7615C475.741 77.7615 475.629 160.707 507 160.707C538.369 160.707 528.629 113 560 113C591.371 113 601.223 1.72803 632.593 1.72803C663.964 1.72803 663.964 139.971 695.335 139.971C726.704 139.971 714.629 70.8494 746 70.8494C777.371 70.8494 789.445 43.2008 820.816 43.2008"
                    stroke="#94ADC7"
                    strokeWidth="3"
                  />
                </g>
                <defs>
                  <linearGradient
                    id="paint0_linear_15_687"
                    x1="-413"
                    y1="0"
                    x2="-413"
                    y2="148"
                    gradientUnits="userSpaceOnUse"
                  >
                    <stop stopColor="#243647" />
                    <stop
                      offset="0.5"
                      stopColor="#243647"
                      stopOpacity="0"
                    />
                  </linearGradient>
                  <clipPath id="clip0_15_687">
                    <rect width="852" height="186" fill="white" />
                  </clipPath>
                </defs>
              </svg>
            </div>
            <div className="chart-months">
              <span>Jan</span>
              <span>Feb</span>
              <span>Mar</span>
              <span>Apr</span>
              <span>May</span>
              <span>Jun</span>
              <span>Jul</span>
            </div>
          </div>
          <div className="actions-section">
            <button className="start-bot-btn">Start Bot</button>
            <button className="stop-bot-btn">Stop Bot</button>
          </div>
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
