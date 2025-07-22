import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import Main from './components/Main';
import Dashboard from './components/Dashboard';
import Overview from './components/Overview';
import Portfolio from './components/Portfolio';
import TradingHistory from './components/TradingHistory';
import Intro from './components/Intro';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Main />} />
        <Route path="/Dashboard" element={<Dashboard />} />
        <Route path="/Overview" element={<Overview />} />
        <Route path="/Portfolio" element={<Portfolio />} />
        <Route path="/TradingHistory" element={<TradingHistory />} />
        <Route path="/Intro" element={<Intro />} />
        {/*
        <Route path="/history" element={<TradingHistory />} />
        <Route path="/portfolio" element={<Portfolio />} />
        */}
      </Routes>
    </BrowserRouter>
  );
}

export default App;
