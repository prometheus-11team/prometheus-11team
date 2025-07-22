import React from "react";
import "./Main.css";
import Header from "./Header";

const Main = () => {
  return (
    <div className="main-container">
      <div className="app-background">
        <Header />

        <main className="main-content">
          <div className="content-wrapper">
            <div className="welcome-section">
              <h2 className="welcome-title">
                Welcome to AMERICAN STOCK HUNTERS
              </h2>
            </div>

            <form className="trading-form">
              <div className="form-field">
                <div className="field-container">
                  <div className="field-header">
                    <label className="field-label">Name</label>
                  </div>
                  <div className="input-container">
                    <input
                      type="text"
                      className="form-input"
                      placeholder="Enter your name"
                    />
                  </div>
                </div>
              </div>

              <div className="form-field">
                <div className="field-container">
                  <div className="field-header">
                    <label className="field-label">Initial Capital</label>
                  </div>
                  <div className="input-container">
                    <input
                      type="text"
                      className="form-input"
                      placeholder="Enter initial capital"
                    />
                  </div>
                </div>
              </div>

              <div className="form-field period-field">
                <div className="date-field">
                  <div className="field-header">
                    <label className="field-label">Investment Period</label>
                  </div>
                  <div className="input-container">
                    <input
                      type="text"
                      className="form-input"
                      placeholder="Start Date"
                    />
                  </div>
                </div>
                <div className="date-field">
                  <div className="field-header">
                    <label className="field-label">End Date</label>
                  </div>
                  <div className="input-container end-date">
                    <input
                      type="text"
                      className="form-input"
                      placeholder="End Date"
                    />
                  </div>
                </div>
              </div>

              <button className="trading-start-btn">Trading Start !</button>
            </form>
          </div>
        </main>
      </div>
    </div>
  );
};

export default Main;
