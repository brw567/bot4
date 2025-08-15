# Complete MCP Setup Guide for Claude Code CLI

## Current Status
✅ Filesystem MCP server is connected and working
❓ GitHub MCP server needs configuration (optional)

## 1. Adding GitHub MCP Server (Optional but Recommended)

### Step 1: Create a GitHub Personal Access Token
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name it: "Claude Code MCP"
4. Select scopes:
   - `repo` (full control of private repositories)
   - `workflow` (if you want to manage GitHub Actions)
   - `read:org` (read org and team membership)
5. Click "Generate token"
6. Copy the token immediately (you won't see it again)

### Step 2: Add GitHub MCP Server
```bash
# Add GitHub MCP server with your token
claude mcp add github npx @modelcontextprotocol/server-github

# When prompted for environment variables, add:
# GITHUB_TOKEN=your_github_token_here
```

### Step 3: Verify Connection
```bash
claude mcp list
# Should show both filesystem and github as ✓ Connected
```

## 2. Project-Specific MCP Configuration

The `.mcp.json` file in your project root allows you to define project-specific MCP servers that will be automatically loaded when you work in this directory.

### Current Project Configuration
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "@modelcontextprotocol/server-filesystem",
        "/home/hamster/bot4"
      ]
    }
  }
}
```

## 3. Available MCP Servers

### Core Servers (Most Useful)
```bash
# Filesystem access (already configured)
claude mcp add filesystem npx @modelcontextprotocol/server-filesystem /home/hamster

# GitHub integration
claude mcp add github npx @modelcontextprotocol/server-github

# PostgreSQL database
claude mcp add postgres npx @modelcontextprotocol/server-postgres postgresql://user:pass@localhost/dbname

# Slack integration
claude mcp add slack npx @modelcontextprotocol/server-slack
```

### Additional Servers
```bash
# Google Drive
npm install -g @modelcontextprotocol/server-gdrive
claude mcp add gdrive npx @modelcontextprotocol/server-gdrive

# Memory/knowledge base
npm install -g @modelcontextprotocol/server-memory
claude mcp add memory npx @modelcontextprotocol/server-memory

# Puppeteer for web automation
npm install -g @modelcontextprotocol/server-puppeteer
claude mcp add puppeteer npx @modelcontextprotocol/server-puppeteer
```

## 4. Using MCP Tools in Claude Code

Once configured, MCP tools are available with the `mcp__` prefix:

### GitHub Tools (when configured)
- `mcp__github__create_repository` - Create a new repo
- `mcp__github__get_file_contents` - Read files from repos
- `mcp__github__push_files` - Push files to a repo
- `mcp__github__create_pull_request` - Create PRs
- `mcp__github__create_issue` - Create issues
- `mcp__github__search_repositories` - Search repos
- `mcp__github__fork_repository` - Fork repos
- And many more...

### Example Usage in Conversation
```
"Create a new GitHub issue for the performance regression in the ML pipeline"
"Push the validated bot4 code to the GitHub repository"
"Search GitHub for similar trading bot implementations"
```

## 5. Environment Variables for MCP

Create a `.env.mcp` file for sensitive configurations:

```bash
cat > /home/hamster/bot4/.env.mcp << 'EOF'
# MCP Configuration
GITHUB_TOKEN=ghp_your_token_here
SLACK_BOT_TOKEN=xoxb-your-slack-token
POSTGRES_CONNECTION_STRING=postgresql://bot4user:bot4pass@localhost:5432/bot4db

# MCP Timeouts
MCP_TIMEOUT=30000
MCP_TOOL_TIMEOUT=60000
EOF

# Add to .gitignore
echo ".env.mcp" >> .gitignore
```

## 6. Troubleshooting MCP

### Check MCP Status
```bash
# List all servers and their status
claude mcp list

# Debug mode for detailed logs
claude --mcp-debug

# Check specific server
/mcp
```

### Common Issues and Fixes

#### Server Not Connecting
```bash
# Remove and re-add the server
claude mcp remove github
claude mcp add github npx @modelcontextprotocol/server-github

# Check npm packages
npm list -g | grep @modelcontextprotocol
```

#### Permission Issues
```bash
# Reset project MCP choices
claude mcp reset-project-choices

# Check project trust
cat .claude.json | grep "hasTrustDialogAccepted"
```

#### Timeout Issues
```bash
# Increase timeouts via environment
export MCP_TIMEOUT=60000
export MCP_TOOL_TIMEOUT=120000
```

## 7. Advanced MCP Configuration

### Custom MCP Server
Create a custom MCP server for bot4-specific operations:

```javascript
// .claude/mcp-servers/bot4-server.js
const { Server } = require('@modelcontextprotocol/sdk');

const server = new Server({
  name: 'bot4-trading',
  version: '1.0.0',
  tools: [
    {
      name: 'validate_strategy',
      description: 'Validate trading strategy configuration',
      inputSchema: {
        type: 'object',
        properties: {
          strategy: { type: 'string' }
        }
      },
      handler: async ({ strategy }) => {
        // Custom validation logic
        return { valid: true, message: 'Strategy validated' };
      }
    }
  ]
});

server.start();
```

### Add Custom Server
```bash
claude mcp add bot4 node /home/hamster/bot4/.claude/mcp-servers/bot4-server.js
```

## 8. MCP Best Practices for Bot4

### 1. Use Project-Scoped Servers
Keep trading-specific MCP servers in `.mcp.json` for the project.

### 2. Secure Credentials
Never commit tokens to Git. Use environment variables or `.env.mcp`.

### 3. Tool Permissions
Configure allowed tools in `.claude/settings.json`:
```json
{
  "allowedTools": [
    "mcp__github__*",
    "mcp__filesystem__*"
  ]
}
```

### 4. Regular Health Checks
```bash
# Add to daily routine
alias mcp-check='claude mcp list'
```

## 9. Integration with Bot4 Workflow

### Development Workflow
```bash
# 1. Start with MCP status check
claude mcp list

# 2. Use GitHub integration for version control
# "Create a PR for today's risk limit updates"

# 3. Use filesystem for rapid development
# "Read all strategy files and validate risk parameters"

# 4. Database integration for config management
# "Check the current trading parameters in PostgreSQL"
```

### Deployment Workflow
```bash
# Use MCP tools in deployment scripts
cat > scripts/mcp_deploy.sh << 'EOF'
#!/bin/bash
# MCP-enhanced deployment

echo "Using MCP to validate deployment..."
claude -p "Use GitHub MCP to check PR status for bot4 repo"
claude -p "Use filesystem MCP to verify all files are ready"
EOF
```

## 10. Quick Reference

### Essential MCP Commands
```bash
claude mcp list              # Show all servers
claude mcp add <name> <cmd>  # Add new server
claude mcp remove <name>     # Remove server
claude mcp get <name>        # Get server details
/mcp                         # Show MCP tools in session
```

### Status Indicators
- ✓ Connected - Server is working
- ✗ Failed - Server connection failed
- ⟳ Connecting - Server is starting up
- ⚠ Timeout - Server took too long to start

## Success Checklist

- [ ] Filesystem MCP server connected
- [ ] GitHub MCP server configured (optional)
- [ ] Project `.mcp.json` created
- [ ] Environment variables secured
- [ ] Tool permissions configured
- [ ] Health check alias created

Your MCP setup will be complete when you can:
1. Access local files via MCP
2. Interact with GitHub repositories
3. Use project-specific tools
4. Have secure credential management

Remember: MCP extends Claude's capabilities significantly. Use it to automate repetitive tasks and integrate with external services.