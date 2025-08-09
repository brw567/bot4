import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  PlusIcon,
  PencilIcon,
  TrashIcon,
  CheckIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { Switch } from '@headlessui/react';
import { AlertRule } from './index';

interface AlertRulesProps {
  rules: AlertRule[];
  onUpdateRule: (rule: AlertRule) => void;
  onDeleteRule: (ruleId: string) => void;
}

const AlertRules: React.FC<AlertRulesProps> = ({ rules, onUpdateRule, onDeleteRule }) => {
  const [editingRule, setEditingRule] = useState<AlertRule | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);

  const metrics = [
    { value: 'cpu_usage', label: 'CPU Usage', unit: '%' },
    { value: 'memory_usage', label: 'Memory Usage', unit: '%' },
    { value: 'win_rate', label: 'Win Rate', unit: '%' },
    { value: 'api_latency', label: 'API Latency', unit: 'ms' },
    { value: 'failed_trades', label: 'Failed Trades', unit: 'count' },
    { value: 'account_balance', label: 'Account Balance', unit: '$' },
    { value: 'trade_volume', label: 'Trade Volume', unit: '$' },
    { value: 'error_rate', label: 'Error Rate', unit: '%' },
  ];

  const operators = [
    { value: '>', label: 'Greater than' },
    { value: '<', label: 'Less than' },
    { value: '>=', label: 'Greater or equal' },
    { value: '<=', label: 'Less or equal' },
    { value: '==', label: 'Equal to' },
    { value: '!=', label: 'Not equal to' },
  ];

  const handleCreateRule = () => {
    const newRule: AlertRule = {
      id: `rule-${Date.now()}`,
      name: '',
      description: '',
      category: 'system',
      condition: {
        metric: 'cpu_usage',
        operator: '>',
        value: 80,
      },
      severity: 'warning',
      enabled: true,
      notifications: ['dashboard'],
      triggerCount: 0,
    };
    setEditingRule(newRule);
    setShowCreateForm(true);
  };

  const handleSaveRule = () => {
    if (editingRule) {
      onUpdateRule(editingRule);
      setEditingRule(null);
      setShowCreateForm(false);
    }
  };

  const handleCancelEdit = () => {
    setEditingRule(null);
    setShowCreateForm(false);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-600 bg-red-100 dark:bg-red-900';
      case 'warning':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900';
      case 'info':
        return 'text-blue-600 bg-blue-100 dark:bg-blue-900';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'system':
        return '🖥️';
      case 'trading':
        return '📈';
      case 'performance':
        return '⚡';
      case 'security':
        return '🔒';
      default:
        return '📌';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">Alert Rules</h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Configure conditions that trigger alerts
          </p>
        </div>
        <button
          onClick={handleCreateRule}
          className="flex items-center px-4 py-2 bg-primary-600 text-white text-sm font-medium rounded-md hover:bg-primary-700 transition-colors"
        >
          <PlusIcon className="h-4 w-4 mr-2" />
          Create Rule
        </button>
      </div>

      {/* Rule Form */}
      {(showCreateForm || editingRule) && editingRule && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="metric-card border-2 border-primary-500"
        >
          <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            {showCreateForm ? 'Create New Rule' : 'Edit Rule'}
          </h4>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Rule Name
              </label>
              <input
                type="text"
                value={editingRule.name}
                onChange={(e) => setEditingRule({ ...editingRule, name: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                placeholder="e.g., High CPU Usage"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Category
              </label>
              <select
                value={editingRule.category}
                onChange={(e) => setEditingRule({ ...editingRule, category: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
              >
                <option value="system">System</option>
                <option value="trading">Trading</option>
                <option value="performance">Performance</option>
                <option value="security">Security</option>
              </select>
            </div>

            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Description
              </label>
              <input
                type="text"
                value={editingRule.description}
                onChange={(e) => setEditingRule({ ...editingRule, description: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                placeholder="Describe what this rule monitors"
              />
            </div>

            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Condition
              </label>
              <div className="flex items-center space-x-2">
                <select
                  value={editingRule.condition.metric}
                  onChange={(e) => setEditingRule({
                    ...editingRule,
                    condition: { ...editingRule.condition, metric: e.target.value }
                  })}
                  className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                >
                  {metrics.map(metric => (
                    <option key={metric.value} value={metric.value}>
                      {metric.label}
                    </option>
                  ))}
                </select>

                <select
                  value={editingRule.condition.operator}
                  onChange={(e) => setEditingRule({
                    ...editingRule,
                    condition: { ...editingRule.condition, operator: e.target.value }
                  })}
                  className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                >
                  {operators.map(op => (
                    <option key={op.value} value={op.value}>
                      {op.label}
                    </option>
                  ))}
                </select>

                <input
                  type="number"
                  value={editingRule.condition.value}
                  onChange={(e) => setEditingRule({
                    ...editingRule,
                    condition: { ...editingRule.condition, value: parseFloat(e.target.value) }
                  })}
                  className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                />

                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {metrics.find(m => m.value === editingRule.condition.metric)?.unit}
                </span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Severity
              </label>
              <select
                value={editingRule.severity}
                onChange={(e) => setEditingRule({ ...editingRule, severity: e.target.value as any })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
              >
                <option value="info">Info</option>
                <option value="warning">Warning</option>
                <option value="critical">Critical</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Duration (minutes)
              </label>
              <input
                type="number"
                value={editingRule.condition.duration || ''}
                onChange={(e) => setEditingRule({
                  ...editingRule,
                  condition: { ...editingRule.condition, duration: parseInt(e.target.value) || undefined }
                })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                placeholder="Optional"
              />
            </div>

            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Notification Channels
              </label>
              <div className="flex flex-wrap gap-2">
                {['email', 'telegram', 'dashboard', 'webhook'].map(channel => (
                  <label key={channel} className="inline-flex items-center">
                    <input
                      type="checkbox"
                      checked={editingRule.notifications.includes(channel)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setEditingRule({
                            ...editingRule,
                            notifications: [...editingRule.notifications, channel]
                          });
                        } else {
                          setEditingRule({
                            ...editingRule,
                            notifications: editingRule.notifications.filter(n => n !== channel)
                          });
                        }
                      }}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="ml-2 text-sm text-gray-700 dark:text-gray-300 capitalize">
                      {channel}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-6 flex justify-end space-x-3">
            <button
              onClick={handleCancelEdit}
              className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm font-medium rounded-md hover:bg-gray-300 dark:hover:bg-gray-600"
            >
              Cancel
            </button>
            <button
              onClick={handleSaveRule}
              className="px-4 py-2 bg-primary-600 text-white text-sm font-medium rounded-md hover:bg-primary-700"
            >
              {showCreateForm ? 'Create' : 'Save'}
            </button>
          </div>
        </motion.div>
      )}

      {/* Rules List */}
      <div className="space-y-4">
        {rules.map((rule) => (
          <motion.div
            key={rule.id}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="metric-card"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-3">
                  <span className="text-2xl">{getCategoryIcon(rule.category)}</span>
                  <div>
                    <h4 className="text-lg font-medium text-gray-900 dark:text-white">
                      {rule.name}
                    </h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {rule.description}
                    </p>
                  </div>
                </div>

                <div className="mt-4 flex flex-wrap items-center gap-4 text-sm">
                  <div className="flex items-center space-x-2">
                    <span className="text-gray-500 dark:text-gray-400">Condition:</span>
                    <code className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded">
                      {rule.condition.metric} {rule.condition.operator} {rule.condition.value}
                      {rule.condition.duration && ` for ${rule.condition.duration}m`}
                    </code>
                  </div>

                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${getSeverityColor(rule.severity)}`}>
                    {rule.severity.toUpperCase()}
                  </span>

                  <div className="flex items-center space-x-1">
                    <span className="text-gray-500 dark:text-gray-400">Notifications:</span>
                    {rule.notifications.map(n => (
                      <span key={n} className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded">
                        {n}
                      </span>
                    ))}
                  </div>
                </div>

                {rule.lastTriggered && (
                  <div className="mt-3 text-sm text-gray-500 dark:text-gray-400">
                    Last triggered: {new Date(rule.lastTriggered).toLocaleString()} • 
                    Total triggers: {rule.triggerCount}
                  </div>
                )}
              </div>

              <div className="flex items-center space-x-3">
                <Switch
                  checked={rule.enabled}
                  onChange={(enabled) => onUpdateRule({ ...rule, enabled })}
                  className={`${
                    rule.enabled ? 'bg-primary-600' : 'bg-gray-200 dark:bg-gray-700'
                  } relative inline-flex h-6 w-11 items-center rounded-full transition-colors`}
                >
                  <span
                    className={`${
                      rule.enabled ? 'translate-x-6' : 'translate-x-1'
                    } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
                  />
                </Switch>

                <button
                  onClick={() => {
                    setEditingRule(rule);
                    setShowCreateForm(false);
                  }}
                  className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                >
                  <PencilIcon className="h-4 w-4" />
                </button>

                <button
                  onClick={() => onDeleteRule(rule.id)}
                  className="p-2 text-gray-400 hover:text-red-600"
                >
                  <TrashIcon className="h-4 w-4" />
                </button>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default AlertRules;