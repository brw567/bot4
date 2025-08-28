#!/usr/bin/env node
/**
 * MCP Server Wrapper for Bot4 Coordinator
 * Bridges Claude CLI to Docker container via stdio
 */

const { spawn } = require('child_process');
const readline = require('readline');

// Check if Docker container is running
const checkContainer = spawn('docker', ['ps', '--filter', 'name=mcp-coordinator', '--format', '{{.Names}}']);
let containerRunning = false;

checkContainer.stdout.on('data', (data) => {
  if (data.toString().includes('mcp-coordinator')) {
    containerRunning = true;
  }
});

checkContainer.on('close', () => {
  if (!containerRunning) {
    console.error('Error: mcp-coordinator container is not running');
    console.error('Start it with: docker-compose -f /home/hamster/bot4/mcp/docker-compose.yml up -d coordinator');
    process.exit(1);
  }

  // Forward stdio to/from Docker container
  const docker = spawn('docker', ['exec', '-i', 'mcp-coordinator', '/app/coordinator']);

  // Forward stdin to Docker
  process.stdin.pipe(docker.stdin);

  // Forward Docker stdout to stdout
  docker.stdout.pipe(process.stdout);

  // Forward Docker stderr to stderr
  docker.stderr.pipe(process.stderr);

  // Handle container exit
  docker.on('exit', (code) => {
    process.exit(code);
  });

  // Handle errors
  docker.on('error', (err) => {
    console.error('Docker exec error:', err);
    process.exit(1);
  });
});