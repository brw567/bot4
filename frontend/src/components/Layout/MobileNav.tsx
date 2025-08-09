import { NavLink } from 'react-router-dom';
import {
  ChartBarIcon,
  TableCellsIcon,
  ChartPieIcon,
  ArrowTrendingUpIcon,
  SparklesIcon,
  BellAlertIcon,
  CpuChipIcon,
  WalletIcon,
} from '@heroicons/react/24/outline';
import { useAppSelector } from '../../hooks/redux';

const MobileNav: React.FC = () => {
  const { unreadCount } = useAppSelector(state => state.alerts);

  const navigation = [
    { name: 'Dashboard', href: '/', icon: ChartBarIcon, shortName: 'Home' },
    { name: 'Portfolio', href: '/portfolio', icon: WalletIcon, shortName: 'Wallet' },
    { name: 'Analytics', href: '/analytics', icon: ChartPieIcon, shortName: 'Charts' },
    { name: 'Trading', href: '/trading', icon: ArrowTrendingUpIcon, shortName: 'Trade' },
    { name: 'Alerts', href: '/alerts', icon: BellAlertIcon, badge: unreadCount, shortName: 'Alerts' },
  ];

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 lg:hidden z-50">
      <div className="grid grid-cols-5 h-16">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              `flex flex-col items-center justify-center space-y-1 relative ${
                isActive
                  ? 'text-primary-600 dark:text-primary-400'
                  : 'text-gray-500 dark:text-gray-400'
              }`
            }
          >
            <div className="relative">
              <item.icon className="h-6 w-6" />
              {item.badge && item.badge > 0 && (
                <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-4 w-4 flex items-center justify-center">
                  {item.badge > 9 ? '9+' : item.badge}
                </span>
              )}
            </div>
            <span className="text-xs">{item.shortName}</span>
          </NavLink>
        ))}
      </div>
    </nav>
  );
};

export default MobileNav;