import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Provider } from 'react-redux';
import { Toaster } from 'react-hot-toast';
import { store } from './store';
import { useAppSelector } from './hooks/redux';
import wsService from './services/websocket';
import InstallPrompt from './components/PWA/InstallPrompt';
import useMetricsPolling from './hooks/useMetricsPolling';

// Layout
import DashboardLayout from './components/Layout/DashboardLayout';

// Pages
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import MetricsPage from './pages/MetricsPage';
import AnalyticsPage from './pages/AnalyticsPage';
import TradingPage from './pages/TradingPage';
import AlertsPage from './pages/AlertsPage';
import SystemHealthPage from './pages/SystemHealthPage';
import AIIntegrationPage from './pages/AIIntegrationPage';
import PortfolioPage from './pages/PortfolioPage';
import ConfigurationPage from './pages/ConfigurationPage';

// Styles
import './index.css';

function AppRoutes() {
  const isAuthenticated = useAppSelector(state => state.auth.isAuthenticated);
  
  // Poll for metrics when authenticated
  useMetricsPolling(5000); // Poll every 5 seconds

  useEffect(() => {
    if (isAuthenticated) {
      // Connect to WebSocket when authenticated
      wsService.connect();
    } else {
      // Disconnect when not authenticated
      wsService.disconnect();
    }
  }, [isAuthenticated]);

  return (
    <Routes>
      <Route path="/login" element={!isAuthenticated ? <LoginPage /> : <Navigate to="/" />} />
      <Route
        path="/"
        element={isAuthenticated ? <DashboardLayout /> : <Navigate to="/login" />}
      >
        <Route index element={<DashboardPage />} />
        <Route path="portfolio" element={<PortfolioPage />} />
        <Route path="metrics" element={<MetricsPage />} />
        <Route path="analytics" element={<AnalyticsPage />} />
        <Route path="trading" element={<TradingPage />} />
        <Route path="ai" element={<AIIntegrationPage />} />
        <Route path="alerts" element={<AlertsPage />} />
        <Route path="system" element={<SystemHealthPage />} />
        <Route path="configuration" element={<ConfigurationPage />} />
      </Route>
    </Routes>
  );
}

function App() {
  return (
    <Provider store={store}>
      <Router>
        <AppRoutes />
        <InstallPrompt />
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#1f2937',
              color: '#fff',
            },
          }}
        />
      </Router>
    </Provider>
  );
}

export default App;
