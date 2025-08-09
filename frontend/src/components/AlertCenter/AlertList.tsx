import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BellIcon,
  XMarkIcon,
  CheckIcon,
  FunnelIcon,
  TrashIcon,
  EyeIcon,
} from '@heroicons/react/24/outline';
import {
  ExclamationTriangleIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  XCircleIcon,
} from '@heroicons/react/24/solid';
import { Alert } from './index';

interface AlertListProps {
  alerts: Alert[];
  selectedAlert: Alert | null;
  onSelectAlert: (alert: Alert | null) => void;
  onMarkAsRead: (alertId: string) => void;
  onMarkAllAsRead: () => void;
  onDelete: (alertId: string) => void;
  onAction: (alert: Alert, action: string) => void;
}

const AlertList: React.FC<AlertListProps> = ({
  alerts,
  selectedAlert,
  onSelectAlert,
  onMarkAsRead,
  onMarkAllAsRead,
  onDelete,
  onAction,
}) => {
  const [filter, setFilter] = useState({
    type: 'all',
    category: 'all',
    unreadOnly: false,
  });

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'critical':
        return <XCircleIcon className="h-5 w-5 text-red-600" />;
      case 'warning':
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-600" />;
      case 'info':
        return <InformationCircleIcon className="h-5 w-5 text-blue-600" />;
      case 'success':
        return <CheckCircleIcon className="h-5 w-5 text-green-600" />;
      default:
        return <BellIcon className="h-5 w-5 text-gray-600" />;
    }
  };

  const getAlertBg = (type: string, read: boolean) => {
    const opacity = read ? '10' : '20';
    switch (type) {
      case 'critical':
        return `bg-red-50 dark:bg-red-900/${opacity} border-red-200 dark:border-red-800`;
      case 'warning':
        return `bg-yellow-50 dark:bg-yellow-900/${opacity} border-yellow-200 dark:border-yellow-800`;
      case 'info':
        return `bg-blue-50 dark:bg-blue-900/${opacity} border-blue-200 dark:border-blue-800`;
      case 'success':
        return `bg-green-50 dark:bg-green-900/${opacity} border-green-200 dark:border-green-800`;
      default:
        return `bg-gray-50 dark:bg-gray-900/${opacity} border-gray-200 dark:border-gray-800`;
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    if (filter.type !== 'all' && alert.type !== filter.type) return false;
    if (filter.category !== 'all' && alert.category !== filter.category) return false;
    if (filter.unreadOnly && alert.read) return false;
    return true;
  });

  const groupedAlerts = filteredAlerts.reduce((acc, alert) => {
    const date = new Date(alert.timestamp).toLocaleDateString();
    if (!acc[date]) acc[date] = [];
    acc[date].push(alert);
    return acc;
  }, {} as { [key: string]: Alert[] });

  return (
    <div className="space-y-6">
      {/* Filters and Actions */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 flex gap-2">
          <select
            value={filter.type}
            onChange={(e) => setFilter({ ...filter, type: e.target.value })}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
          >
            <option value="all">All Types</option>
            <option value="critical">Critical</option>
            <option value="warning">Warning</option>
            <option value="info">Info</option>
            <option value="success">Success</option>
          </select>

          <select
            value={filter.category}
            onChange={(e) => setFilter({ ...filter, category: e.target.value })}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
          >
            <option value="all">All Categories</option>
            <option value="system">System</option>
            <option value="trading">Trading</option>
            <option value="performance">Performance</option>
            <option value="security">Security</option>
          </select>

          <button
            onClick={() => setFilter({ ...filter, unreadOnly: !filter.unreadOnly })}
            className={`px-3 py-2 text-sm font-medium rounded-md transition-colors ${
              filter.unreadOnly
                ? 'bg-primary-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            Unread Only
          </button>
        </div>

        <div className="flex gap-2">
          <button
            onClick={onMarkAllAsRead}
            className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm font-medium rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          >
            <CheckIcon className="h-4 w-4 inline mr-1" />
            Mark All Read
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Alert List */}
        <div className="lg:col-span-2 space-y-4">
          {Object.keys(groupedAlerts).length === 0 ? (
            <div className="text-center py-12 metric-card">
              <BellIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500 dark:text-gray-400">No alerts found</p>
            </div>
          ) : (
            Object.entries(groupedAlerts).map(([date, dateAlerts]) => (
              <div key={date}>
                <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                  {date === new Date().toLocaleDateString() ? 'Today' : date}
                </h3>
                <div className="space-y-2">
                  <AnimatePresence>
                    {dateAlerts.map((alert) => (
                      <motion.div
                        key={alert.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, x: -100 }}
                        className={`p-4 rounded-lg border cursor-pointer transition-all ${
                          getAlertBg(alert.type, alert.read)
                        } ${selectedAlert?.id === alert.id ? 'ring-2 ring-primary-500' : ''} ${
                          !alert.read ? 'font-medium' : ''
                        }`}
                        onClick={() => onSelectAlert(alert)}
                      >
                        <div className="flex items-start space-x-3">
                          {getAlertIcon(alert.type)}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between">
                              <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                                {alert.title}
                              </h4>
                              <span className="text-xs text-gray-500 dark:text-gray-400">
                                {new Date(alert.timestamp).toLocaleTimeString([], {
                                  hour: '2-digit',
                                  minute: '2-digit',
                                })}
                              </span>
                            </div>
                            <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
                              {alert.message}
                            </p>
                            <div className="mt-2 flex items-center space-x-2">
                              <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded">
                                {alert.category}
                              </span>
                              {!alert.read && (
                                <span className="text-xs text-primary-600 dark:text-primary-400">
                                  • New
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            {!alert.read && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onMarkAsRead(alert.id);
                                }}
                                className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                                title="Mark as read"
                              >
                                <EyeIcon className="h-4 w-4" />
                              </button>
                            )}
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                onDelete(alert.id);
                              }}
                              className="p-1 text-gray-400 hover:text-red-600"
                              title="Delete"
                            >
                              <TrashIcon className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Alert Details */}
        <div className="lg:col-span-1">
          <div className="metric-card sticky top-6">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
              Alert Details
            </h3>
            {selectedAlert ? (
              <div className="space-y-4">
                <div>
                  <div className="flex items-center space-x-2 mb-2">
                    {getAlertIcon(selectedAlert.type)}
                    <h4 className="font-medium text-gray-900 dark:text-white">
                      {selectedAlert.title}
                    </h4>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
                    {selectedAlert.message}
                  </p>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Type</span>
                    <span className={`font-medium capitalize ${
                      selectedAlert.type === 'critical' ? 'text-red-600' :
                      selectedAlert.type === 'warning' ? 'text-yellow-600' :
                      selectedAlert.type === 'info' ? 'text-blue-600' :
                      'text-green-600'
                    }`}>
                      {selectedAlert.type}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Category</span>
                    <span className="font-medium text-gray-900 dark:text-white capitalize">
                      {selectedAlert.category}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Time</span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {new Date(selectedAlert.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Status</span>
                    <span className={`font-medium ${
                      selectedAlert.read ? 'text-gray-600' : 'text-primary-600'
                    }`}>
                      {selectedAlert.read ? 'Read' : 'Unread'}
                    </span>
                  </div>
                </div>

                {selectedAlert.metadata && (
                  <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                    <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Additional Information
                    </h5>
                    <pre className="text-xs bg-gray-100 dark:bg-gray-800 p-3 rounded overflow-x-auto">
                      {JSON.stringify(selectedAlert.metadata, null, 2)}
                    </pre>
                  </div>
                )}

                {selectedAlert.actionable && selectedAlert.actions && (
                  <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                    <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                      Actions
                    </h5>
                    <div className="space-y-2">
                      {selectedAlert.actions.map((action, index) => (
                        <button
                          key={index}
                          onClick={() => onAction(selectedAlert, action.action)}
                          className="w-full px-4 py-2 bg-primary-600 text-white text-sm font-medium rounded-md hover:bg-primary-700 transition-colors"
                        >
                          {action.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8">
                <BellIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Select an alert to view details
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlertList;