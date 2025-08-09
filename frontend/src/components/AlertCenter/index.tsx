import React, { useState, useEffect } from 'react';
import { useAppSelector, useAppDispatch } from '../../hooks/redux';
import AlertList from './AlertList';
import AlertRules from './AlertRules';
import AlertHistory from './AlertHistory';
import NotificationSettings from './NotificationSettings';
import { markAsRead, markAllAsRead, deleteAlert } from '../../store/slices/alertsSlice';
import {
  BellIcon,
  Cog6ToothIcon,
  ClockIcon,
  ShieldCheckIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import { BellAlertIcon } from '@heroicons/react/24/solid';

export interface Alert {
  id: string;
  type: 'critical' | 'warning' | 'info' | 'success';
  category: 'system' | 'trading' | 'performance' | 'security';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actionable: boolean;
  actions?: {
    label: string;
    action: string;
  }[];
  metadata?: any;
}

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  category: string;
  condition: {
    metric: string;
    operator: string;
    value: number;
    duration?: number;
  };
  severity: 'critical' | 'warning' | 'info';
  enabled: boolean;
  notifications: string[];
  lastTriggered?: string;
  triggerCount: number;
}

const AlertCenter: React.FC = () => {
  const dispatch = useAppDispatch();
  const { alerts, unreadCount } = useAppSelector(state => state.alerts);
  const [activeTab, setActiveTab] = useState('active');
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  
  // Mock data
  const [alertRules, setAlertRules] = useState<AlertRule[]>([]);
  const [notificationChannels, setNotificationChannels] = useState<any[]>([]);

  useEffect(() => {
    // Generate mock alert rules
    const rules: AlertRule[] = [
      {
        id: 'rule-1',
        name: 'High CPU Usage',
        description: 'Alert when CPU usage exceeds 80% for 5 minutes',
        category: 'system',
        condition: {
          metric: 'cpu_usage',
          operator: '>',
          value: 80,
          duration: 5,
        },
        severity: 'warning',
        enabled: true,
        notifications: ['email', 'dashboard'],
        lastTriggered: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
        triggerCount: 3,
      },
      {
        id: 'rule-2',
        name: 'Win Rate Drop',
        description: 'Alert when win rate drops below 80%',
        category: 'trading',
        condition: {
          metric: 'win_rate',
          operator: '<',
          value: 80,
        },
        severity: 'critical',
        enabled: true,
        notifications: ['email', 'telegram', 'dashboard'],
        lastTriggered: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        triggerCount: 1,
      },
      {
        id: 'rule-3',
        name: 'API Latency',
        description: 'Alert when API latency exceeds 500ms',
        category: 'performance',
        condition: {
          metric: 'api_latency',
          operator: '>',
          value: 500,
        },
        severity: 'warning',
        enabled: true,
        notifications: ['dashboard'],
        triggerCount: 5,
      },
      {
        id: 'rule-4',
        name: 'Failed Trades',
        description: 'Alert on consecutive failed trades',
        category: 'trading',
        condition: {
          metric: 'failed_trades',
          operator: '>',
          value: 3,
        },
        severity: 'critical',
        enabled: true,
        notifications: ['email', 'telegram'],
        lastTriggered: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
        triggerCount: 2,
      },
      {
        id: 'rule-5',
        name: 'Low Balance',
        description: 'Alert when account balance drops below threshold',
        category: 'trading',
        condition: {
          metric: 'account_balance',
          operator: '<',
          value: 1000,
        },
        severity: 'warning',
        enabled: false,
        notifications: ['email'],
        triggerCount: 0,
      },
    ];
    setAlertRules(rules);

    // Generate notification channels
    const channels = [
      {
        id: 'email',
        name: 'Email',
        type: 'email',
        config: { address: 'trader@example.com' },
        enabled: true,
        icon: 'EnvelopeIcon',
      },
      {
        id: 'telegram',
        name: 'Telegram',
        type: 'telegram',
        config: { chatId: '@trading_alerts' },
        enabled: true,
        icon: 'ChatBubbleLeftIcon',
      },
      {
        id: 'dashboard',
        name: 'Dashboard',
        type: 'dashboard',
        config: {},
        enabled: true,
        icon: 'ComputerDesktopIcon',
      },
      {
        id: 'webhook',
        name: 'Webhook',
        type: 'webhook',
        config: { url: 'https://api.example.com/alerts' },
        enabled: false,
        icon: 'GlobeAltIcon',
      },
    ];
    setNotificationChannels(channels);
  }, []);

  const tabs = [
    { id: 'active', name: 'Active Alerts', icon: BellIcon, count: unreadCount },
    { id: 'rules', name: 'Alert Rules', icon: ShieldCheckIcon },
    { id: 'history', name: 'Alert History', icon: ClockIcon },
    { id: 'settings', name: 'Notifications', icon: Cog6ToothIcon },
  ];

  const handleAlertAction = (alert: Alert, action: string) => {
    console.log(`Executing action ${action} for alert ${alert.id}`);
    // Handle alert actions
  };

  const handleDeleteAlert = (alertId: string) => {
    dispatch(deleteAlert(alertId));
  };

  const handleMarkAsRead = (alertId: string) => {
    dispatch(markAsRead(alertId));
  };

  const handleMarkAllAsRead = () => {
    dispatch(markAllAsRead());
  };

  const activeAlerts = alerts.filter(a => !a.read);
  const criticalAlerts = alerts.filter(a => a.type === 'critical' && !a.read);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Alert Center</h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Manage alerts, rules, and notification preferences
          </p>
        </div>
        {criticalAlerts.length > 0 && (
          <div className="flex items-center space-x-2 px-4 py-2 bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-200 rounded-lg">
            <ExclamationTriangleIcon className="h-5 w-5" />
            <span className="text-sm font-medium">
              {criticalAlerts.length} Critical Alert{criticalAlerts.length > 1 ? 's' : ''}
            </span>
          </div>
        )}
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Active Alerts</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {activeAlerts.length}
              </p>
            </div>
            <BellAlertIcon className="h-8 w-8 text-yellow-600" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Critical</p>
              <p className="text-2xl font-semibold text-red-600">
                {criticalAlerts.length}
              </p>
            </div>
            <ExclamationTriangleIcon className="h-8 w-8 text-red-600" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Active Rules</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {alertRules.filter(r => r.enabled).length}
              </p>
            </div>
            <ShieldCheckIcon className="h-8 w-8 text-green-600" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Channels</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {notificationChannels.filter(c => c.enabled).length}
              </p>
            </div>
            <Cog6ToothIcon className="h-8 w-8 text-blue-600" />
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                flex items-center py-2 px-1 border-b-2 font-medium text-sm transition-colors
                ${activeTab === tab.id
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                }
              `}
            >
              <tab.icon className="h-5 w-5 mr-2" />
              {tab.name}
              {tab.count && tab.count > 0 && (
                <span className="ml-2 bg-red-500 text-white text-xs px-2 py-0.5 rounded-full">
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="mt-6">
        {activeTab === 'active' && (
          <AlertList
            alerts={alerts}
            selectedAlert={selectedAlert}
            onSelectAlert={setSelectedAlert}
            onMarkAsRead={handleMarkAsRead}
            onMarkAllAsRead={handleMarkAllAsRead}
            onDelete={handleDeleteAlert}
            onAction={handleAlertAction}
          />
        )}
        {activeTab === 'rules' && (
          <AlertRules
            rules={alertRules}
            onUpdateRule={(rule) => {
              setAlertRules(prev => prev.map(r => r.id === rule.id ? rule : r));
            }}
            onDeleteRule={(ruleId) => {
              setAlertRules(prev => prev.filter(r => r.id !== ruleId));
            }}
          />
        )}
        {activeTab === 'history' && (
          <AlertHistory alerts={alerts.filter(a => a.read)} />
        )}
        {activeTab === 'settings' && (
          <NotificationSettings
            channels={notificationChannels}
            onUpdateChannel={(channel) => {
              setNotificationChannels(prev => prev.map(c => c.id === channel.id ? channel : c));
            }}
          />
        )}
      </div>
    </div>
  );
};

export default AlertCenter;