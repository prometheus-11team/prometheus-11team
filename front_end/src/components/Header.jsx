import React from "react";
import { Link } from "react-router-dom";

const Header = () => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo-section">
          <img
            className="logo-image"
            src="https://api.builder.io/api/v1/image/assets/TEMP/358c8a24857f988331c14db54789fc990f06d8c7?width=100"
            alt="America Stock Hunters Logo"
          />
          <div className="logo-text-container">
            <Link to="/" className="company-name" style={{ textDecoration: 'none', color: '#fff' }}>
              AMERICAN STOCK HUNTERS
            </Link>
          </div>
        </div>
        <div className="navigation-section">
          <nav className="navigation-menu">
            <Link className="nav-item" to="/dashboard">Dashboard</Link>
            <Link className="nav-item" to="/TradingHistory">Trading History</Link>
            <Link className="nav-item" to="/Intro">Introduction</Link>
            <Link className="nav-item" to="/Overview">Overview</Link>
            <Link className="nav-item" to="/Portfolio">Portfolio</Link>
          </nav>
          <img
            className="user-avatar"
            src="https://api.builder.io/api/v1/image/assets/TEMP/eb71d8841e2be4901cdc561db634ef4587cd378a?width=80"
            alt="User Avatar"
          />
        </div>
      </div>
    </header>
  );
};

export default Header; 