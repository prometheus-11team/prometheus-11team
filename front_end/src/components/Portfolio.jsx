import React from "react";
import {
  PieChart,
  Pie,
  Cell,
  Legend,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import "./Portfolio.css";
import Header from "./Header";

const M7_DATA = [
  { name: "AAPL", value: 20 },
  { name: "MSFT", value: 18 },
  { name: "TSLA", value: 15 },
  { name: "NVDA", value: 22 },
  { name: "AMZN", value: 9 },
  { name: "GOOGL", value: 8 },
  { name: "META", value: 8 },
];
const COLORS = [
  "#00A76A",
  "#0072E3",
  "#FFB300",
  "#FF5630",
  "#36B37E",
  "#6554C0",
  "#FF6F61",
];

const HOLDINGS = [
  { symbol: "AAPL", qty: 15, buy: 180.0, now: 195.2, profit: 8.4, total: 2928, daily: 0.7 },
  { symbol: "MSFT", qty: 12, buy: 320.0, now: 315.0, profit: -1.6, total: 3780, daily: -0.2 },
  { symbol: "TSLA", qty: 8, buy: 250.0, now: 265.0, profit: 6.0, total: 2120, daily: 0.5 },
  { symbol: "NVDA", qty: 20, buy: 900.0, now: 950.0, profit: 5.6, total: 19000, daily: 1.1 },
  { symbol: "AMZN", qty: 5, buy: 130.0, now: 128.0, profit: -1.5, total: 640, daily: -0.1 },
  { symbol: "GOOGL", qty: 4, buy: 140.0, now: 145.0, profit: 3.6, total: 580, daily: 0.2 },
  { symbol: "META", qty: 3, buy: 300.0, now: 310.0, profit: 3.3, total: 930, daily: 0.3 },
];

const Portfolio = () => {
  return (
    <div className="portfolio-container">
      <div className="portfolio-background">
        <div className="portfolio-layout">
          <Header />

          {/* Main Portfolio Content */}
          <div className="portfolio-main">
            <div className="portfolio-content">
              <div className="portfolio-widgets-container">
                {/* Portfolio Donut Chart */}
                <div className="portfolio-chart-widget">
                  <div className="portfolio-widget-title">
                    포트폴리오 비중 그래프
                  </div>
                  {/* 높이를 340 → 400 으로 키웠습니다. */}
                  <ResponsiveContainer width="100%" height={400}>
                    <PieChart>
                      <Pie
                        data={M7_DATA}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        /* innerRadius, outerRadius를 좀 더 키웠습니다 */
                        innerRadius={90}
                        outerRadius={150}
                        fill="#8884d8"
                        label={({ name, percent }) =>
                          `${name} ${(percent * 100).toFixed(1)}%`
                        }
                        stroke="#fff"
                        strokeWidth={2}
                      >
                        {M7_DATA.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={COLORS[index % COLORS.length]}
                          />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value, name) => [`${value}%`, name]} />
                      {/* 범례 텍스트 크기를 키우기 위해 wrapperStyle 추가 */}
                      <Legend
                        verticalAlign="middle"
                        align="right"
                        layout="vertical"
                        iconType="circle"
                        wrapperStyle={{ fontSize: "16px" }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                {/* Portfolio Holdings Table */}
                <div className="portfolio-table-widget">
                  <div className="portfolio-widget-title">
                    보유 종목 상세 리스트
                  </div>
                  <div className="portfolio-table-container">
                    <div className="portfolio-table-header">
                      <div className="header-cell col-symbol">종목</div>
                      <div className="header-cell col-qty">보유 수량</div>
                      <div className="header-cell col-buy">매입가</div>
                      <div className="header-cell col-current">현재가</div>
                      <div className="header-cell col-profit">수익률</div>
                      <div className="header-cell col-total">총 금액</div>
                      <div className="header-cell col-daily">일간 수익률</div>
                    </div>
                    {HOLDINGS.map((row) => (
                      <div
                        className="portfolio-table-row"
                        key={row.symbol}
                      >
                        <div className="row-cell col-symbol symbol-cell">
                          {row.symbol}
                        </div>
                        <div className="row-cell col-qty">
                          {row.qty}주
                        </div>
                        <div className="row-cell col-buy">
                          {row.buy.toFixed(1)}
                        </div>
                        <div className="row-cell col-current">
                          {row.now.toFixed(1)}
                        </div>
                        {/* profit-positive/negative 클래스 적용은 CSS에서 색상을 바꿔주세요 */}
                        <div
                          className={`row-cell col-profit ${
                            row.profit >= 0
                              ? "profit-positive"
                              : "profit-negative"
                          }`}
                        >
                          {row.profit >= 0 ? "+" : ""}
                          {row.profit.toFixed(1)}%
                        </div>
                        <div className="row-cell col-total">
                          {row.total}
                        </div>
                        <div
                          className={`row-cell col-daily ${
                            row.daily >= 0
                              ? "profit-positive"
                              : "profit-negative"
                          }`}
                        >
                          {row.daily >= 0 ? "+" : ""}
                          {row.daily.toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              {/* 거래내역 및 사이드바 메트릭스 완전히 제거됨 */}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Portfolio;
