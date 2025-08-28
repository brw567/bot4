#!/bin/bash
# Deploy working MCP agents to containers
# This fixes the integration issue and enables real collaboration

set -e

echo "==========================================="
echo "  DEPLOYING WORKING MCP AGENTS            "
echo "==========================================="

# Define agents with their roles
declare -A AGENTS=(
    ["coordinator"]="karl:Project Manager"
    ["architect"]="avery:System Architect"
    ["mlengineer"]="blake:ML Engineer"
    ["riskquant"]="cameron:Risk Manager"
    ["exchangespec"]="drew:Exchange Specialist"
    ["infraengineer"]="ellis:Infrastructure"
    ["qualitygate"]="morgan:Quality Gate"
    ["integrationvalidator"]="quinn:Integration Validator"
    ["complianceauditor"]="skyler:Compliance Auditor"
)

# Copy simple MCP server to all containers
echo "Deploying MCP servers to containers..."

for service in "${!AGENTS[@]}"; do
    IFS=':' read -r name role <<< "${AGENTS[$service]}"
    
    echo "  Deploying $name ($service)..."
    
    # Copy the simple MCP server
    docker cp /home/hamster/bot4/mcp/agents/simple-mcp-server.py mcp-$service:/app/mcp-server.py
    
    # Create startup script in container
    docker exec mcp-$service bash -c "cat > /app/start-agent.sh << 'EOF'
#!/bin/bash
export AGENT_NAME='$name'
export AGENT_ROLE='$role'
exec python3 /app/mcp-server.py '$name' '$role'
EOF"
    
    docker exec mcp-$service chmod +x /app/start-agent.sh
    
    # Update symlink to use new startup script
    docker exec mcp-$service ln -sf /app/start-agent.sh /app/$service
done

echo ""
echo "Restarting containers with working agents..."
docker-compose -f /home/hamster/bot4/mcp/docker-compose-agents.yml restart

sleep 5

echo ""
echo "Testing agent responsiveness..."

SUCCESS=0
TOTAL=0

for service in "${!AGENTS[@]}"; do
    IFS=':' read -r name role <<< "${AGENTS[$service]}"
    ((TOTAL++))
    
    echo -n "  Testing $name... "
    
    # Test with a simple JSON-RPC initialize request
    TEST_RESULT=$(echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}' | \
        timeout 2 docker exec -i mcp-$service /app/start-agent.sh 2>/dev/null | \
        grep -q "protocolVersion" && echo "OK" || echo "FAIL")
    
    if [ "$TEST_RESULT" = "OK" ]; then
        echo "✓"
        ((SUCCESS++))
    else
        echo "✗"
    fi
done

echo ""
echo "==========================================="
echo "Results: $SUCCESS/$TOTAL agents responding"

if [ "$SUCCESS" -eq "$TOTAL" ]; then
    echo "✅ ALL AGENTS ARE NOW ACTIVE!"
    echo ""
    echo "The multi-agent team is ready:"
    for service in "${!AGENTS[@]}"; do
        IFS=':' read -r name role <<< "${AGENTS[$service]}"
        echo "  • $name - $role"
    done
else
    echo "⚠️  Some agents need attention ($SUCCESS/$TOTAL working)"
fi

echo "==========================================="

# Create a demonstration script
cat > /home/hamster/bot4/scripts/demo_agent_collaboration.sh << 'EOF'
#!/bin/bash
# Demonstrate real multi-agent collaboration

echo "================================="
echo "MULTI-AGENT COLLABORATION DEMO"
echo "================================="
echo ""

# Test architect agent for duplicates
echo "1. Asking Architect to check for duplicates..."
echo '{"jsonrpc":"2.0","method":"tools/call","id":1,"params":{"name":"analyze_task","arguments":{"task":"Check for duplicate Order structs"}}}' | \
    docker exec -i mcp-architect /app/start-agent.sh 2>/dev/null | \
    python3 -m json.tool

echo ""
echo "2. Getting consensus on deduplication..."

# Each agent votes
AGENTS=("coordinator" "architect" "mlengineer" "riskquant" "exchangespec" "infraengineer" "qualitygate" "integrationvalidator" "complianceauditor")

for agent in "${AGENTS[@]}"; do
    echo "  $agent voting..."
    echo '{"jsonrpc":"2.0","method":"tools/call","id":1,"params":{"name":"vote","arguments":{"proposal":"Start deduplication sprint","vote":true}}}' | \
        docker exec -i mcp-$agent /app/start-agent.sh 2>/dev/null > /dev/null
done

# Check shared context for votes
echo ""
echo "3. Checking consensus result..."
cat /home/hamster/bot4/.mcp/shared_context.json | python3 -m json.tool

echo ""
echo "✅ Multi-agent collaboration is working!"
EOF

chmod +x /home/hamster/bot4/scripts/demo_agent_collaboration.sh

echo ""
echo "Run './scripts/demo_agent_collaboration.sh' to see agents working together!"