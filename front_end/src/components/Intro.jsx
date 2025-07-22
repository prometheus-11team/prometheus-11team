import React from "react";
import "./Intro.css";
import Header from "./Header";

const Intro = () => {
  return (
    <div className="intro-container">
      <Header />
      <div className="intro-background">
        <div className="intro-layout">
          {/* Main Content Area */}
          <div className="intro-main">
            <div className="intro-content">
              <div className="content-wrapper">
                {/* Title Section */}
                <div className="title-section">
                  <div className="title-header">
                    <div className="title-container">
                      <h1 className="main-title" style={{ whiteSpace: 'nowrap' }}>
                        Stock Price Prediction and Trading Pipeline
                      </h1>
                    </div>
                  </div>

                  {/* Feature Cards Section */}
                  <div className="features-section">
                    <div className="feature-card">
                      <div className="card-content">
                        <h3 className="card-title">Auto Trading System</h3>
                      </div>
                    </div>

                    <div className="feature-card">
                      <div className="card-content">
                        <h3 className="card-title">
                          Stock Price Prediction Model
                        </h3>
                      </div>
                      <div className="card-spacer"></div>
                    </div>
                  </div>

                  {/* Action Buttons Section */}
                  <div className="actions-section">
                    <div className="actions-container">
                      <button className="primary-action-btn">대단해요~</button>
                      <button className="secondary-action-btn">
                        응원해요~
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Intro;
