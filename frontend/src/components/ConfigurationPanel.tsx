import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Switch,
  Button,
  TextField,
  Grid,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Tabs,
  Tab,
  FormControlLabel,
  Select,
  MenuItem,
  InputAdornment,
  Tooltip,
  CircularProgress
} from '@mui/material';
import {
  Settings,
  Check,
  Close,
  Visibility,
  VisibilityOff,
  TestTube,
  Save,
  CloudUpload,
  CloudDownload,
  Security,
  Warning
} from '@mui/icons-material';

interface ExchangeConfig {
  id: string;
  name: string;
  enabled: boolean;
  apiKey: string;
  apiSecret: string;
  passphrase?: string;
  testnet: boolean;
  features: {
    spotTrading: boolean;
    futuresTrading: boolean;
    marginTrading: boolean;
    websocket: boolean;
  };
  rateLimits: {
    requestsPerSecond: number;
    ordersPerSecond: number;
  };
  feeTier: string;
  status: 'connected' | 'disconnected' | 'error' | 'testing';
  balance?: number;
  lastTested?: Date;
}

interface DataSourceConfig {
  id: string;
  name: string;
  enabled: boolean;
  apiKey: string;
  apiSecret?: string;
  endpoint?: string;
  rateLimit: number;
  status: 'connected' | 'disconnected' | 'error';
}

interface CommunicationConfig {
  id: string;
  name: string;
  enabled: boolean;
  config: Record<string, any>;
  status: 'connected' | 'disconnected' | 'error';
}

const ConfigurationPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [exchanges, setExchanges] = useState<ExchangeConfig[]>([
    {
      id: 'binance',
      name: 'Binance',
      enabled: false,
      apiKey: '',
      apiSecret: '',
      testnet: true,
      features: {
        spotTrading: true,
        futuresTrading: false,
        marginTrading: false,
        websocket: true
      },
      rateLimits: {
        requestsPerSecond: 10,
        ordersPerSecond: 5
      },
      feeTier: 'VIP0',
      status: 'disconnected'
    },
    {
      id: 'coinbase',
      name: 'Coinbase',
      enabled: false,
      apiKey: '',
      apiSecret: '',
      testnet: true,
      features: {
        spotTrading: true,
        futuresTrading: false,
        marginTrading: false,
        websocket: true
      },
      rateLimits: {
        requestsPerSecond: 10,
        ordersPerSecond: 3
      },
      feeTier: 'Standard',
      status: 'disconnected'
    },
    {
      id: 'okx',
      name: 'OKX',
      enabled: false,
      apiKey: '',
      apiSecret: '',
      passphrase: '',
      testnet: true,
      features: {
        spotTrading: true,
        futuresTrading: true,
        marginTrading: true,
        websocket: true
      },
      rateLimits: {
        requestsPerSecond: 20,
        ordersPerSecond: 10
      },
      feeTier: 'VIP1',
      status: 'disconnected'
    },
    {
      id: 'bybit',
      name: 'Bybit',
      enabled: false,
      apiKey: '',
      apiSecret: '',
      testnet: true,
      features: {
        spotTrading: true,
        futuresTrading: true,
        marginTrading: false,
        websocket: true
      },
      rateLimits: {
        requestsPerSecond: 50,
        ordersPerSecond: 10
      },
      feeTier: 'None',
      status: 'disconnected'
    },
    {
      id: 'kraken',
      name: 'Kraken',
      enabled: false,
      apiKey: '',
      apiSecret: '',
      testnet: false,
      features: {
        spotTrading: true,
        futuresTrading: true,
        marginTrading: true,
        websocket: true
      },
      rateLimits: {
        requestsPerSecond: 15,
        ordersPerSecond: 5
      },
      feeTier: 'Starter',
      status: 'disconnected'
    }
  ]);

  const [dataSources, setDataSources] = useState<DataSourceConfig[]>([
    {
      id: 'coingecko',
      name: 'CoinGecko',
      enabled: false,
      apiKey: '',
      rateLimit: 50,
      status: 'disconnected'
    },
    {
      id: 'tradingview',
      name: 'TradingView',
      enabled: false,
      apiKey: '',
      rateLimit: 100,
      status: 'disconnected'
    },
    {
      id: 'coinmarketcap',
      name: 'CoinMarketCap',
      enabled: false,
      apiKey: '',
      rateLimit: 333,
      status: 'disconnected'
    }
  ]);

  const [communications, setCommunications] = useState<CommunicationConfig[]>([
    {
      id: 'xai',
      name: 'xAI (Grok)',
      enabled: false,
      config: {
        apiKey: '',
        model: 'grok-1',
        maxTokens: 1000
      },
      status: 'disconnected'
    },
    {
      id: 'telegram',
      name: 'Telegram Bot',
      enabled: false,
      config: {
        botToken: '',
        channelId: '',
        enableAlerts: true,
        enableReports: true
      },
      status: 'disconnected'
    },
    {
      id: 'discord',
      name: 'Discord',
      enabled: false,
      config: {
        webhookUrl: '',
        enableAlerts: true
      },
      status: 'disconnected'
    }
  ]);

  const [selectedExchange, setSelectedExchange] = useState<ExchangeConfig | null>(null);
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({});
  const [testingInProgress, setTestingInProgress] = useState<Record<string, boolean>>({});
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  const handleExchangeToggle = (exchangeId: string) => {
    setExchanges(prev => prev.map(ex => 
      ex.id === exchangeId ? { ...ex, enabled: !ex.enabled } : ex
    ));
  };

  const handleExchangeConfigure = (exchange: ExchangeConfig) => {
    setSelectedExchange(exchange);
    setConfigDialogOpen(true);
  };

  const handleTestConnection = async (exchangeId: string) => {
    setTestingInProgress({ ...testingInProgress, [exchangeId]: true });
    
    // Simulate API call
    try {
      const response = await fetch(`http://192.168.100.64:8000/api/test-exchange/${exchangeId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(exchanges.find(ex => ex.id === exchangeId))
      });
      
      if (response.ok) {
        setExchanges(prev => prev.map(ex => 
          ex.id === exchangeId 
            ? { ...ex, status: 'connected', lastTested: new Date() } 
            : ex
        ));
      } else {
        throw new Error('Connection failed');
      }
    } catch (error) {
      setExchanges(prev => prev.map(ex => 
        ex.id === exchangeId ? { ...ex, status: 'error' } : ex
      ));
    } finally {
      setTestingInProgress({ ...testingInProgress, [exchangeId]: false });
    }
  };

  const handleSaveConfiguration = async () => {
    setSaveStatus('saving');
    
    try {
      const config = {
        exchanges,
        dataSources,
        communications,
        timestamp: new Date().toISOString()
      };
      
      const response = await fetch('http://192.168.100.64:8000/api/configuration', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      
      if (response.ok) {
        setSaveStatus('saved');
        setTimeout(() => setSaveStatus('idle'), 3000);
      } else {
        throw new Error('Save failed');
      }
    } catch (error) {
      setSaveStatus('error');
      setTimeout(() => setSaveStatus('idle'), 3000);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'success';
      case 'disconnected': return 'default';
      case 'error': return 'error';
      case 'testing': return 'warning';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return '🟢';
      case 'disconnected': return '⚫';
      case 'error': return '🔴';
      case 'testing': return '🟡';
      default: return '⚫';
    }
  };

  return (
    <Box sx={{ width: '100%', p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Bot3 Trading Platform - Configuration
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        Configure your exchange API keys, data sources, and communication channels. 
        All credentials are encrypted and stored securely. Start with testnet/sandbox environments.
      </Alert>

      <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)} sx={{ mb: 3 }}>
        <Tab label="Exchanges" />
        <Tab label="Data Sources" />
        <Tab label="AI & Communication" />
        <Tab label="Security" />
      </Tabs>

      {activeTab === 0 && (
        <Grid container spacing={2}>
          {exchanges.map((exchange) => (
            <Grid item xs={12} md={6} key={exchange.id}>
              <Card>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Box display="flex" alignItems="center" gap={2}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={exchange.enabled}
                            onChange={() => handleExchangeToggle(exchange.id)}
                            color="primary"
                          />
                        }
                        label={exchange.name}
                      />
                      {exchange.testnet && (
                        <Chip label="TESTNET" size="small" color="warning" />
                      )}
                    </Box>
                    <Box display="flex" gap={1}>
                      <Tooltip title={`Status: ${exchange.status}`}>
                        <span>{getStatusIcon(exchange.status)}</span>
                      </Tooltip>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => handleExchangeConfigure(exchange)}
                        startIcon={<Settings />}
                      >
                        Configure
                      </Button>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => handleTestConnection(exchange.id)}
                        disabled={testingInProgress[exchange.id]}
                        startIcon={testingInProgress[exchange.id] ? 
                          <CircularProgress size={16} /> : <TestTube />}
                      >
                        Test
                      </Button>
                    </Box>
                  </Box>
                  
                  {exchange.apiKey && (
                    <Box mt={2}>
                      <Typography variant="caption" color="textSecondary">
                        API Key: {exchange.apiKey.substring(0, 8)}...
                      </Typography>
                      {exchange.lastTested && (
                        <Typography variant="caption" color="textSecondary" display="block">
                          Last tested: {new Date(exchange.lastTested).toLocaleString()}
                        </Typography>
                      )}
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {activeTab === 1 && (
        <Grid container spacing={2}>
          {dataSources.map((source) => (
            <Grid item xs={12} md={6} key={source.id}>
              <Card>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <FormControlLabel
                      control={
                        <Switch
                          checked={source.enabled}
                          onChange={() => {
                            setDataSources(prev => prev.map(s => 
                              s.id === source.id ? { ...s, enabled: !s.enabled } : s
                            ));
                          }}
                          color="primary"
                        />
                      }
                      label={source.name}
                    />
                    <Box display="flex" gap={1}>
                      <Tooltip title={`Status: ${source.status}`}>
                        <span>{getStatusIcon(source.status)}</span>
                      </Tooltip>
                      <Button size="small" variant="outlined" startIcon={<Settings />}>
                        Configure
                      </Button>
                    </Box>
                  </Box>
                  <Typography variant="caption" color="textSecondary">
                    Rate Limit: {source.rateLimit} req/min
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {activeTab === 2 && (
        <Grid container spacing={2}>
          {communications.map((comm) => (
            <Grid item xs={12} md={6} key={comm.id}>
              <Card>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <FormControlLabel
                      control={
                        <Switch
                          checked={comm.enabled}
                          onChange={() => {
                            setCommunications(prev => prev.map(c => 
                              c.id === comm.id ? { ...c, enabled: !c.enabled } : c
                            ));
                          }}
                          color="primary"
                        />
                      }
                      label={comm.name}
                    />
                    <Box display="flex" gap={1}>
                      <Tooltip title={`Status: ${comm.status}`}>
                        <span>{getStatusIcon(comm.status)}</span>
                      </Tooltip>
                      <Button size="small" variant="outlined" startIcon={<Settings />}>
                        Configure
                      </Button>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {activeTab === 3 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Security Settings
            </Typography>
            <Alert severity="warning" sx={{ mb: 2 }}>
              <Typography variant="subtitle2">Important Security Notes:</Typography>
              <ul>
                <li>Never enable withdrawal permissions on API keys</li>
                <li>Use IP whitelisting when available</li>
                <li>Enable 2FA on all exchange accounts</li>
                <li>Rotate API keys every 90 days</li>
                <li>Always test in sandbox/testnet first</li>
              </ul>
            </Alert>
            
            <Box display="flex" gap={2} mt={3}>
              <Button variant="contained" startIcon={<CloudUpload />}>
                Import Configuration
              </Button>
              <Button variant="contained" startIcon={<CloudDownload />}>
                Export Configuration
              </Button>
              <Button variant="outlined" color="warning" startIcon={<Security />}>
                Rotate All Keys
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Configuration Dialog */}
      <Dialog open={configDialogOpen} onClose={() => setConfigDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          Configure {selectedExchange?.name}
        </DialogTitle>
        <DialogContent>
          {selectedExchange && (
            <Box sx={{ pt: 2 }}>
              <TextField
                fullWidth
                label="API Key"
                value={selectedExchange.apiKey}
                onChange={(e) => {
                  setSelectedExchange({ ...selectedExchange, apiKey: e.target.value });
                }}
                margin="normal"
              />
              <TextField
                fullWidth
                label="API Secret"
                type={showSecrets[selectedExchange.id] ? 'text' : 'password'}
                value={selectedExchange.apiSecret}
                onChange={(e) => {
                  setSelectedExchange({ ...selectedExchange, apiSecret: e.target.value });
                }}
                margin="normal"
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={() => setShowSecrets({
                          ...showSecrets,
                          [selectedExchange.id]: !showSecrets[selectedExchange.id]
                        })}
                      >
                        {showSecrets[selectedExchange.id] ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  )
                }}
              />
              {selectedExchange.passphrase !== undefined && (
                <TextField
                  fullWidth
                  label="Passphrase (if required)"
                  type="password"
                  value={selectedExchange.passphrase}
                  onChange={(e) => {
                    setSelectedExchange({ ...selectedExchange, passphrase: e.target.value });
                  }}
                  margin="normal"
                />
              )}
              
              <Typography variant="subtitle2" sx={{ mt: 3, mb: 1 }}>
                Features
              </Typography>
              <Box>
                <FormControlLabel
                  control={
                    <Switch
                      checked={selectedExchange.features.spotTrading}
                      onChange={(e) => {
                        setSelectedExchange({
                          ...selectedExchange,
                          features: {
                            ...selectedExchange.features,
                            spotTrading: e.target.checked
                          }
                        });
                      }}
                    />
                  }
                  label="Spot Trading"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={selectedExchange.features.futuresTrading}
                      onChange={(e) => {
                        setSelectedExchange({
                          ...selectedExchange,
                          features: {
                            ...selectedExchange.features,
                            futuresTrading: e.target.checked
                          }
                        });
                      }}
                    />
                  }
                  label="Futures Trading"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={selectedExchange.features.websocket}
                      onChange={(e) => {
                        setSelectedExchange({
                          ...selectedExchange,
                          features: {
                            ...selectedExchange.features,
                            websocket: e.target.checked
                          }
                        });
                      }}
                    />
                  }
                  label="WebSocket Streams"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={selectedExchange.testnet}
                      onChange={(e) => {
                        setSelectedExchange({
                          ...selectedExchange,
                          testnet: e.target.checked
                        });
                      }}
                    />
                  }
                  label="Use Testnet/Sandbox"
                />
              </Box>

              <Typography variant="subtitle2" sx={{ mt: 3, mb: 1 }}>
                Rate Limits
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Requests/sec"
                    type="number"
                    value={selectedExchange.rateLimits.requestsPerSecond}
                    onChange={(e) => {
                      setSelectedExchange({
                        ...selectedExchange,
                        rateLimits: {
                          ...selectedExchange.rateLimits,
                          requestsPerSecond: parseInt(e.target.value)
                        }
                      });
                    }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Orders/sec"
                    type="number"
                    value={selectedExchange.rateLimits.ordersPerSecond}
                    onChange={(e) => {
                      setSelectedExchange({
                        ...selectedExchange,
                        rateLimits: {
                          ...selectedExchange.rateLimits,
                          ordersPerSecond: parseInt(e.target.value)
                        }
                      });
                    }}
                  />
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfigDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={() => handleTestConnection(selectedExchange!.id)}
            disabled={testingInProgress[selectedExchange?.id || '']}
          >
            Test Connection
          </Button>
          <Button 
            variant="contained" 
            onClick={() => {
              setExchanges(prev => prev.map(ex => 
                ex.id === selectedExchange?.id ? selectedExchange : ex
              ));
              setConfigDialogOpen(false);
            }}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* Save Status Bar */}
      <Box position="fixed" bottom={20} right={20}>
        <Button
          variant="contained"
          color={saveStatus === 'saved' ? 'success' : 'primary'}
          onClick={handleSaveConfiguration}
          disabled={saveStatus === 'saving'}
          startIcon={
            saveStatus === 'saving' ? <CircularProgress size={20} /> :
            saveStatus === 'saved' ? <Check /> :
            saveStatus === 'error' ? <Close /> :
            <Save />
          }
        >
          {saveStatus === 'saving' ? 'Saving...' :
           saveStatus === 'saved' ? 'Saved!' :
           saveStatus === 'error' ? 'Error!' :
           'Save All Configuration'}
        </Button>
      </Box>
    </Box>
  );
};

export default ConfigurationPanel;