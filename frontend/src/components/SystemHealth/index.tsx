import React, { useState, useEffect } from 'react';
import { useAppSelector } from '../../hooks/redux';
import ResourceMonitor from './ResourceMonitor';
import ServiceStatus from './ServiceStatus';
import PerformanceMetrics from './PerformanceMetrics';
import SystemLogs from './SystemLogs';
import {
  CpuChipIcon,
  ServerIcon,
  ChartBarIcon,
  DocumentTextIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';

const SystemHealth: React.FC = () => {
  const [activeTab, setActiveTab] = useState('resources');
  const { health, botStatus, connected } = useAppSelector(state => state.system);
  const [systemData, setSystemData] = useState<any>({
    resources: [],
    services: {},
    performance: {},
    logs: [],
  });
  const [realSystemStatus, setRealSystemStatus] = useState<any>(null);
  const [apiCounters, setApiCounters] = useState<any>(null);
  const [apiHealth, setApiHealth] = useState<any>(null);
  
  // Fetch real system status
  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        const response = await fetch('/api/system/status', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`,
          },
        });
        if (response.ok) {
          const data = await response.json();
          setRealSystemStatus(data.services);
        }
      } catch (error) {
        console.error('Failed to fetch system status:', error);
      }
    };
    
    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 10000); // Update every 10 seconds
    
    return () => clearInterval(interval);
  }, []);
  
  // Fetch API counters
  useEffect(() => {
    const fetchApiCounters = async () => {
      try {
        const response = await fetch('/api/system/api-counters', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`,
          },
        });
        if (response.ok) {
          const data = await response.json();
          setApiCounters(data.counters);
        }
      } catch (error) {
        console.error('Failed to fetch API counters:', error);
      }
    };
    
    fetchApiCounters();
    const interval = setInterval(fetchApiCounters, 30000); // Update every 30 seconds
    
    return () => clearInterval(interval);
  }, []);
  
  // Fetch API health
  useEffect(() => {
    const fetchApiHealth = async () => {
      try {
        const response = await fetch('/api/system/api-health', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`,
          },
        });
        if (response.ok) {
          const data = await response.json();
          setApiHealth(data.health);
        }
      } catch (error) {
        console.error('Failed to fetch API health:', error);
      }
    };
    
    fetchApiHealth();
    const interval = setInterval(fetchApiHealth, 60000); // Update every minute
    
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Fetch real system health data
    const fetchSystemData = async () => {
      const token = localStorage.getItem('token');
      const headers = { 'Authorization': `Bearer ${token}` };

      try {
        // Fetch system metrics
        const metricsResponse = await fetch('/api/system/metrics', { headers });
        const metrics = await metricsResponse.json();

        // Fetch service health
        const servicesResponse = await fetch('/api/system/services', { headers });
        const servicesData = await servicesResponse.json();

        // Fetch performance metrics
        const performanceResponse = await fetch('/api/system/performance', { headers });
        const performance = await performanceResponse.json();

        // Format resources data for chart
        const resources = [{
          timestamp: new Date().toISOString(),
          cpu: metrics.cpu_usage || 0,
          memory: metrics.memory_usage || 0,
          disk: metrics.disk_usage || 0,
          network: (metrics.network_io?.bytes_sent + metrics.network_io?.bytes_recv) / 1024 / 1024 || 0,
          threads: metrics.thread_count || 0,
          connections: metrics.process_count || 0,
        }];

        // Format services data
        const services: any = {};
        if (servicesData.services) {
          servicesData.services.forEach((service: any) => {
            services[service.name] = {
              status: service.status,
              uptime: service.uptime,
              lastCheck: service.last_check,
              cpu: service.cpu,
              memory: service.memory,
              restarts: service.restarts,
            };
          });
        }

        // Add exchange services status from realSystemStatus
        if (realSystemStatus) {
          services['Trading Bot'] = {
            status: realSystemStatus.bot ? 'healthy' : 'stopped',
            uptime: services['python']?.uptime || 'Unknown',
            lastCheck: new Date().toISOString(),
            cpu: services['python']?.cpu || 0,
            memory: services['python']?.memory || 0,
            restarts: 0,
          };
        }

        // System logs will be fetched when logs tab is active
        const logs: any[] = [];

        setSystemData({
          resources,
          services,
          performance,
          logs,
        });
      } catch (error) {
        console.error('Failed to fetch system data:', error);
      }
    };

    fetchSystemData();
    const interval = setInterval(fetchSystemData, 5000);
    return () => clearInterval(interval);
  }, [botStatus, connected, realSystemStatus]);

  const tabs = [
    { id: 'resources', name: 'Resource Monitor', icon: CpuChipIcon },
    { id: 'services', name: 'Service Status', icon: ServerIcon },
    { id: 'performance', name: 'Performance', icon: ChartBarIcon },
    { id: 'logs', name: 'System Logs', icon: DocumentTextIcon },
  ];

  // Calculate system health score
  const healthScore = health ? (
    (100 - health.cpuUsage) * 0.3 +
    (100 - health.memoryUsage) * 0.3 +
    (health.redisConnected ? 100 : 0) * 0.2 +
    (100 - (health.diskUsage || 70)) * 0.2
  ) : 0;

  const getHealthColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getHealthLabel = (score: number) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Critical';
  };

  return (
    <div className="space-y-6">
      {/* Header with Overall Health */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">System Health</h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Monitor and manage system performance
          </p>
        </div>
        <div className="text-center">
          <div className={`text-4xl font-bold ${getHealthColor(healthScore)}`}>
            {healthScore.toFixed(0)}%
          </div>
          <p className={`text-sm font-medium ${getHealthColor(healthScore)}`}>
            {getHealthLabel(healthScore)}
          </p>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">CPU Usage</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {(health?.cpuUsage || 0).toFixed(1)}%
              </p>
            </div>
            <CpuChipIcon className={`h-8 w-8 ${
              health?.cpuUsage > 80 ? 'text-red-600' : 
              health?.cpuUsage > 60 ? 'text-yellow-600' : 
              'text-green-600'
            }`} />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Memory Usage</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {(health?.memoryUsage || 0).toFixed(1)}%
              </p>
            </div>
            <ServerIcon className={`h-8 w-8 ${
              health?.memoryUsage > 80 ? 'text-red-600' : 
              health?.memoryUsage > 60 ? 'text-yellow-600' : 
              'text-green-600'
            }`} />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Active Services</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {Object.values(systemData.services).filter((s: any) => s.status === 'healthy').length}/
                {Object.keys(systemData.services).length}
              </p>
            </div>
            <div className="text-primary-600">
              <ChartBarIcon className="h-8 w-8" />
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">System Alerts</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {systemData.logs.filter((l: any) => l.level === 'error').length}
              </p>
            </div>
            <ExclamationTriangleIcon className={`h-8 w-8 ${
              systemData.logs.filter((l: any) => l.level === 'error').length > 0 
                ? 'text-red-600' 
                : 'text-gray-400'
            }`} />
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
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="mt-6">
        {activeTab === 'resources' && (
          <ResourceMonitor data={systemData.resources} currentHealth={health} />
        )}
        {activeTab === 'services' && (
          <ServiceStatus services={systemData.services} apiLatency={health?.apiLatency} />
        )}
        {activeTab === 'performance' && (
          <PerformanceMetrics metrics={systemData.performance} apiCounters={apiCounters} apiHealth={apiHealth} />
        )}
        {activeTab === 'logs' && (
          <SystemLogs logs={systemData.logs} />
        )}
      </div>
    </div>
  );
};

export default SystemHealth;