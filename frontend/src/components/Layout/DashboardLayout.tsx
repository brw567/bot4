import React, { useState } from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import { useAppSelector, useAppDispatch } from '../../hooks/redux';
import { logout } from '../../store/slices/authSlice';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useBotStatus } from '../../hooks/useBotStatus';
import MobileNav from './MobileNav';
import {
  ChartBarIcon,
  TableCellsIcon,
  ChartPieIcon,
  ArrowTrendingUpIcon,
  SparklesIcon,
  BellAlertIcon,
  CpuChipIcon,
  ArrowRightOnRectangleIcon,
  Bars3Icon,
  XMarkIcon,
  WalletIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';

const DashboardLayout: React.FC = () => {
  const dispatch = useAppDispatch();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { username } = useAppSelector(state => state.auth);
  const { botStatus, connected } = useAppSelector(state => state.system);
  const { winRate, activePairs } = useAppSelector(state => state.metrics);
  const { unreadCount } = useAppSelector(state => state.alerts);

  // Initialize WebSocket connection
  useWebSocket();
  
  // Poll bot status
  useBotStatus();

  const handleLogout = () => {
    dispatch(logout());
  };

  const navigation = [
    { name: 'Dashboard', href: '/', icon: ChartBarIcon },
    { name: 'Portfolio', href: '/portfolio', icon: WalletIcon },
    { name: 'Metrics', href: '/metrics', icon: TableCellsIcon },
    { name: 'Analytics', href: '/analytics', icon: ChartPieIcon },
    { name: 'Trading', href: '/trading', icon: ArrowTrendingUpIcon },
    { name: 'AI Insights', href: '/ai', icon: SparklesIcon },
    { name: 'Alerts', href: '/alerts', icon: BellAlertIcon, badge: unreadCount },
    { name: 'System Health', href: '/system', icon: CpuChipIcon },
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="lg:hidden p-2 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                {mobileMenuOpen ? (
                  <XMarkIcon className="h-6 w-6" />
                ) : (
                  <Bars3Icon className="h-6 w-6" />
                )}
              </button>
              <h1 className="ml-2 lg:ml-0 text-lg lg:text-xl font-semibold text-gray-900 dark:text-white">
                Bot2 Trading System
              </h1>
              <div className="ml-6 hidden lg:flex items-center space-x-4">
                <span className={clsx(
                  'px-2 py-1 text-xs font-medium rounded-full',
                  connected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                )}>
                  {connected ? 'Connected' : 'Disconnected'}
                </span>
                <span className={clsx(
                  'px-2 py-1 text-xs font-medium rounded-full',
                  botStatus === 'running' ? 'bg-green-100 text-green-800' : 
                  botStatus === 'stopped' ? 'bg-yellow-100 text-yellow-800' : 
                  'bg-gray-100 text-gray-800'
                )}>
                  Bot: {botStatus}
                </span>
                <span className="hidden sm:inline text-sm text-gray-600 dark:text-gray-400">
                  Win Rate: <span className="font-medium text-green-600">{winRate.toFixed(1)}%</span>
                </span>
                <span className="hidden md:inline text-sm text-gray-600 dark:text-gray-400">
                  Active Pairs: <span className="font-medium">{activePairs}</span>
                </span>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-700 dark:text-gray-300">
                {username}
              </span>
              <button
                onClick={handleLogout}
                className="flex items-center text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                <ArrowRightOnRectangleIcon className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-4rem)] relative">
        {/* Mobile Menu Overlay */}
        {mobileMenuOpen && (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
            onClick={() => setMobileMenuOpen(false)}
          />
        )}

        {/* Sidebar */}
        <nav className={clsx(
          'fixed lg:static inset-y-0 left-0 z-50 w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 transform transition-transform duration-300 ease-in-out lg:translate-x-0',
          mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
        )}>
          <div className="px-3 py-4">
            <ul className="space-y-1">
              {navigation.map((item) => (
                <li key={item.name}>
                  <NavLink
                    to={item.href}
                    className={({ isActive }) =>
                      clsx(
                        'flex items-center justify-between px-3 py-2 text-sm font-medium rounded-md transition-colors',
                        isActive
                          ? 'bg-primary-100 text-primary-900 dark:bg-primary-900 dark:text-primary-100'
                          : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                      )
                    }
                  >
                    <div className="flex items-center">
                      <item.icon className="h-5 w-5 mr-3" />
                      {item.name}
                    </div>
                    {item.badge && item.badge > 0 && (
                      <span className="bg-red-500 text-white text-xs font-medium px-2 py-0.5 rounded-full">
                        {item.badge}
                      </span>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto pb-16 lg:pb-0">
          <div className="p-4 sm:p-6">
            <Outlet />
          </div>
        </main>
      </div>

      {/* Mobile Bottom Navigation */}
      <MobileNav />
    </div>
  );
};

export default DashboardLayout;