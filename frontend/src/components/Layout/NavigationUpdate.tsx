// Add this to your existing navigation component

import { CompareArrows } from '@mui/icons-material';

// Add to navigation items array:
const multiExchangeNavItem = {
  text: 'Multi-Exchange',
  icon: <CompareArrows />,
  path: '/multi-exchange',
  description: 'Cross-exchange analytics and arbitrage'
};

// Example navigation structure:
export const navigationItems = [
  {
    text: 'Dashboard',
    icon: <Dashboard />,
    path: '/',
  },
  {
    text: 'Trading',
    icon: <ShowChart />,
    path: '/trading',
  },
  {
    text: 'Multi-Exchange',
    icon: <CompareArrows />,
    path: '/multi-exchange',
  },
  {
    text: 'Analytics',
    icon: <Analytics />,
    path: '/analytics',
  },
  {
    text: 'AI Integration',
    icon: <Psychology />,
    path: '/ai-integration',
  },
  {
    text: 'Alerts',
    icon: <Notifications />,
    path: '/alerts',
  },
  {
    text: 'System Health',
    icon: <MonitorHeart />,
    path: '/system-health',
  },
];