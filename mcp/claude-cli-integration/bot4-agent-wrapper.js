#!/usr/bin/env node
/**
 * Universal MCP Agent Wrapper for Bot4
 * Bridges Claude CLI to any Docker-based agent
 */

const { spawn } = require('child_process');
const path = require('path');

// Get agent name from script name or argument
const scriptName = path.basename(process.argv[1]);
const agentName = process.argv[2] || scriptName.replace('bot4-', '').replace('.js', '');

// Map agent names to their Docker container names and ports
const agentConfig = {
  'coordinator': { container: 'mcp-coordinator', port: 8000 },
  'architect': { container: 'mcp-architect', port: 8080 },
  'riskquant': { container: 'mcp-riskquant', port: 8081 },
  'mlengineer': { container: 'mcp-mlengineer', port: 8082 },
  'exchangespec': { container: 'mcp-exchangespec', port: 8083 },
  'infraengineer': { container: 'mcp-infraengineer', port: 8084 },
  'qualitygate': { container: 'mcp-qualitygate', port: 8085 },
  'integrationvalidator': { container: 'mcp-integrationvalidator', port: 8086 },
  'complianceauditor': { container: 'mcp-complianceauditor', port: 8087 }
};

const config = agentConfig[agentName];
if (!config) {
  console.error(`Unknown agent: ${agentName}`);
  console.error('Valid agents:', Object.keys(agentConfig).join(', '));
  process.exit(1);
}

// Check if Docker container is running
const checkContainer = spawn('docker', ['ps', '--filter', `name=${config.container}`, '--format', '{{.Names}}']);
let containerRunning = false;

checkContainer.stdout.on('data', (data) => {
  if (data.toString().includes(config.container)) {
    containerRunning = true;
  }
});

checkContainer.on('close', () => {
  if (!containerRunning) {
    // Try to start the container
    console.error(`Starting ${config.container} container...`);
    const startContainer = spawn('docker-compose', [
      '-f', '/home/hamster/bot4/mcp/docker-compose.yml',
      'up', '-d', agentName
    ], { stdio: 'inherit' });
    
    startContainer.on('close', (code) => {
      if (code !== 0) {
        console.error(`Failed to start ${config.container} container`);
        process.exit(1);
      }
      
      // Wait a moment for container to be ready
      setTimeout(() => {
        connectToContainer();
      }, 3000);
    });
  } else {
    connectToContainer();
  }
});

function connectToContainer() {
  // Use HTTP-to-stdio bridge for MCP protocol
  const http = require('http');
  const { Readable, Writable } = require('stream');
  
  // Create stdin reader
  const rl = require('readline').createInterface({
    input: process.stdin,
    output: null,
    terminal: false
  });
  
  // Process MCP messages from stdin
  rl.on('line', (line) => {
    try {
      const message = JSON.parse(line);
      
      // Forward to container via HTTP
      const postData = JSON.stringify(message);
      const options = {
        hostname: 'localhost',
        port: config.port,
        path: '/mcp',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(postData)
        }
      };
      
      const req = http.request(options, (res) => {
        let data = '';
        res.on('data', (chunk) => {
          data += chunk;
        });
        res.on('end', () => {
          // Send response back to Claude CLI
          process.stdout.write(data + '\n');
        });
      });
      
      req.on('error', (e) => {
        console.error(`Problem with request: ${e.message}`);
      });
      
      req.write(postData);
      req.end();
    } catch (e) {
      console.error('Error parsing MCP message:', e);
    }
  });
  
  // Send initial capability announcement
  const capabilities = {
    jsonrpc: '2.0',
    method: 'initialize',
    params: {
      protocolVersion: '2024-11-05',
      capabilities: {
        tools: {},
        resources: {},
        prompts: {}
      },
      clientInfo: {
        name: `bot4-${agentName}`,
        version: '1.0.0'
      }
    }
  };
  
  process.stdout.write(JSON.stringify(capabilities) + '\n');
}