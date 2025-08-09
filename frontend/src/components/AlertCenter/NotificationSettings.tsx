import React, { useState } from 'react';
import { Switch } from '@headlessui/react';
import {
  EnvelopeIcon,
  ChatBubbleLeftIcon,
  ComputerDesktopIcon,
  GlobeAltIcon,
  BellIcon,
  PlusIcon,
  PencilIcon,
  TrashIcon,
} from '@heroicons/react/24/outline';

interface NotificationChannel {
  id: string;
  name: string;
  type: string;
  config: any;
  enabled: boolean;
  icon: string;
}

interface NotificationSettingsProps {
  channels: NotificationChannel[];
  onUpdateChannel: (channel: NotificationChannel) => void;
}

const NotificationSettings: React.FC<NotificationSettingsProps> = ({ channels, onUpdateChannel }) => {
  const [editingChannel, setEditingChannel] = useState<NotificationChannel | null>(null);
  const [testingChannel, setTestingChannel] = useState<string | null>(null);

  const getIcon = (iconName: string) => {
    const icons: { [key: string]: any } = {
      EnvelopeIcon,
      ChatBubbleLeftIcon,
      ComputerDesktopIcon,
      GlobeAltIcon,
    };
    const Icon = icons[iconName] || BellIcon;
    return <Icon className="h-6 w-6" />;
  };

  const handleTestChannel = async (channelId: string) => {
    setTestingChannel(channelId);
    // Simulate test
    await new Promise(resolve => setTimeout(resolve, 2000));
    setTestingChannel(null);
    // Show success notification
  };

  const channelTypes = [
    {
      type: 'email',
      name: 'Email',
      icon: 'EnvelopeIcon',
      fields: [
        { key: 'address', label: 'Email Address', type: 'email', required: true },
        { key: 'smtpHost', label: 'SMTP Host', type: 'text' },
        { key: 'smtpPort', label: 'SMTP Port', type: 'number' },
      ],
    },
    {
      type: 'telegram',
      name: 'Telegram',
      icon: 'ChatBubbleLeftIcon',
      fields: [
        { key: 'chatId', label: 'Chat ID', type: 'text', required: true },
        { key: 'botToken', label: 'Bot Token', type: 'password' },
      ],
    },
    {
      type: 'webhook',
      name: 'Webhook',
      icon: 'GlobeAltIcon',
      fields: [
        { key: 'url', label: 'Webhook URL', type: 'url', required: true },
        { key: 'method', label: 'HTTP Method', type: 'select', options: ['POST', 'PUT'] },
        { key: 'headers', label: 'Headers (JSON)', type: 'textarea' },
      ],
    },
  ];

  const handleAddChannel = (type: string) => {
    const channelType = channelTypes.find(ct => ct.type === type);
    if (channelType) {
      const newChannel: NotificationChannel = {
        id: `channel-${Date.now()}`,
        name: `New ${channelType.name}`,
        type: channelType.type,
        config: {},
        enabled: false,
        icon: channelType.icon,
      };
      setEditingChannel(newChannel);
    }
  };

  const handleSaveChannel = () => {
    if (editingChannel) {
      onUpdateChannel(editingChannel);
      setEditingChannel(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Channel List */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">Notification Channels</h3>
          <div className="flex items-center space-x-2">
            {channelTypes.map(ct => (
              <button
                key={ct.type}
                onClick={() => handleAddChannel(ct.type)}
                className="flex items-center px-3 py-2 bg-primary-600 text-white text-sm font-medium rounded-md hover:bg-primary-700 transition-colors"
              >
                <PlusIcon className="h-4 w-4 mr-1" />
                {ct.name}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          {channels.map((channel) => (
            <div key={channel.id} className="metric-card">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="p-3 bg-gray-100 dark:bg-gray-800 rounded-lg">
                    {getIcon(channel.icon)}
                  </div>
                  <div>
                    <h4 className="text-lg font-medium text-gray-900 dark:text-white">
                      {channel.name}
                    </h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {channel.type === 'email' && channel.config.address}
                      {channel.type === 'telegram' && channel.config.chatId}
                      {channel.type === 'webhook' && channel.config.url}
                      {channel.type === 'dashboard' && 'In-app notifications'}
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-3">
                  <button
                    onClick={() => handleTestChannel(channel.id)}
                    disabled={testingChannel === channel.id}
                    className="px-3 py-1 text-sm font-medium text-primary-600 hover:text-primary-700 disabled:opacity-50"
                  >
                    {testingChannel === channel.id ? 'Testing...' : 'Test'}
                  </button>

                  <Switch
                    checked={channel.enabled}
                    onChange={(enabled) => onUpdateChannel({ ...channel, enabled })}
                    className={`${
                      channel.enabled ? 'bg-primary-600' : 'bg-gray-200 dark:bg-gray-700'
                    } relative inline-flex h-6 w-11 items-center rounded-full transition-colors`}
                  >
                    <span
                      className={`${
                        channel.enabled ? 'translate-x-6' : 'translate-x-1'
                      } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
                    />
                  </Switch>

                  {channel.type !== 'dashboard' && (
                    <>
                      <button
                        onClick={() => setEditingChannel(channel)}
                        className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                      >
                        <PencilIcon className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => onUpdateChannel({ ...channel, enabled: false })}
                        className="p-2 text-gray-400 hover:text-red-600"
                      >
                        <TrashIcon className="h-4 w-4" />
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Edit Channel Modal */}
      {editingChannel && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
              Configure {channelTypes.find(ct => ct.type === editingChannel.type)?.name} Channel
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Channel Name
                </label>
                <input
                  type="text"
                  value={editingChannel.name}
                  onChange={(e) => setEditingChannel({ ...editingChannel, name: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-900"
                />
              </div>

              {channelTypes
                .find(ct => ct.type === editingChannel.type)
                ?.fields.map(field => (
                  <div key={field.key}>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      {field.label} {field.required && <span className="text-red-500">*</span>}
                    </label>
                    {field.type === 'select' ? (
                      <select
                        value={editingChannel.config[field.key] || ''}
                        onChange={(e) => setEditingChannel({
                          ...editingChannel,
                          config: { ...editingChannel.config, [field.key]: e.target.value }
                        })}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-900"
                      >
                        {field.options?.map(opt => (
                          <option key={opt} value={opt}>{opt}</option>
                        ))}
                      </select>
                    ) : field.type === 'textarea' ? (
                      <textarea
                        value={editingChannel.config[field.key] || ''}
                        onChange={(e) => setEditingChannel({
                          ...editingChannel,
                          config: { ...editingChannel.config, [field.key]: e.target.value }
                        })}
                        rows={3}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-900"
                      />
                    ) : (
                      <input
                        type={field.type}
                        value={editingChannel.config[field.key] || ''}
                        onChange={(e) => setEditingChannel({
                          ...editingChannel,
                          config: { ...editingChannel.config, [field.key]: e.target.value }
                        })}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-900"
                      />
                    )}
                  </div>
                ))}
            </div>

            <div className="mt-6 flex justify-end space-x-3">
              <button
                onClick={() => setEditingChannel(null)}
                className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm font-medium rounded-md hover:bg-gray-300 dark:hover:bg-gray-600"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveChannel}
                className="px-4 py-2 bg-primary-600 text-white text-sm font-medium rounded-md hover:bg-primary-700"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Notification Preferences */}
      <div className="metric-card">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Notification Preferences
        </h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                Critical Alerts
              </h4>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Always notify for critical system issues
              </p>
            </div>
            <Switch
              checked={true}
              onChange={() => {}}
              className="bg-primary-600 relative inline-flex h-6 w-11 items-center rounded-full"
            >
              <span className="translate-x-6 inline-block h-4 w-4 transform rounded-full bg-white" />
            </Switch>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                Alert Grouping
              </h4>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Group similar alerts to reduce noise
              </p>
            </div>
            <Switch
              checked={true}
              onChange={() => {}}
              className="bg-primary-600 relative inline-flex h-6 w-11 items-center rounded-full"
            >
              <span className="translate-x-6 inline-block h-4 w-4 transform rounded-full bg-white" />
            </Switch>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                Quiet Hours
              </h4>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Pause non-critical alerts during specified hours
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="time"
                defaultValue="22:00"
                className="px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-900"
              />
              <span className="text-sm text-gray-500">to</span>
              <input
                type="time"
                defaultValue="08:00"
                className="px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-900"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NotificationSettings;