#!/usr/bin/env python3
"""
Simple MCP Server Implementation
Works without mcp package - implements protocol directly
"""

import json
import sys
import os
from typing import Dict, List, Any, Optional

class SimpleMCPServer:
    """Minimal MCP server implementation"""
    
    def __init__(self, agent_name: str, role: str):
        self.agent_name = agent_name
        self.role = role
        self.shared_context_path = "/home/hamster/bot4/.mcp/shared_context.json"
        
    def handle_initialize(self, request: Dict) -> Dict:
        """Handle initialization request"""
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "0.1.0",
                "capabilities": {
                    "tools": {
                        "listChanged": False
                    }
                },
                "serverInfo": {
                    "name": self.agent_name,
                    "version": "1.0.0"
                }
            }
        }
    
    def handle_list_tools(self, request: Dict) -> Dict:
        """List available tools"""
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": [
                    {
                        "name": "analyze_task",
                        "description": f"{self.agent_name} analyzes a task",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"}
                            },
                            "required": ["task"]
                        }
                    },
                    {
                        "name": "vote",
                        "description": f"{self.agent_name} votes on a proposal",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "proposal": {"type": "string"},
                                "vote": {"type": "boolean"}
                            },
                            "required": ["proposal", "vote"]
                        }
                    },
                    {
                        "name": "get_status",
                        "description": f"Get {self.agent_name}'s status",
                        "inputSchema": {"type": "object"}
                    }
                ]
            }
        }
    
    def handle_call_tool(self, request: Dict) -> Dict:
        """Handle tool calls"""
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        result = {}
        
        if tool_name == "analyze_task":
            task = arguments.get("task", "")
            result = {
                "agent": self.agent_name,
                "role": self.role,
                "task": task,
                "status": "analyzing",
                "message": f"{self.agent_name} is analyzing: {task}"
            }
            # Update shared context
            self.update_context("current_task", task)
            
        elif tool_name == "vote":
            proposal = arguments.get("proposal", "")
            vote = arguments.get("vote", False)
            result = self.record_vote(proposal, vote)
            
        elif tool_name == "get_status":
            result = {
                "agent": self.agent_name,
                "role": self.role,
                "status": "active",
                "ready": True
            }
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
        
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }
                ]
            }
        }
    
    def update_context(self, key: str, value: Any):
        """Update shared context"""
        try:
            context = {}
            if os.path.exists(self.shared_context_path):
                with open(self.shared_context_path, 'r') as f:
                    context = json.load(f)
            
            if 'agents' not in context:
                context['agents'] = {}
            
            if self.agent_name not in context['agents']:
                context['agents'][self.agent_name] = {}
            
            context['agents'][self.agent_name][key] = value
            
            with open(self.shared_context_path, 'w') as f:
                json.dump(context, f, indent=2)
        except Exception as e:
            sys.stderr.write(f"Error updating context: {e}\n")
    
    def record_vote(self, proposal: str, vote: bool) -> Dict:
        """Record a vote in shared context"""
        try:
            context = {}
            if os.path.exists(self.shared_context_path):
                with open(self.shared_context_path, 'r') as f:
                    context = json.load(f)
            
            if 'votes' not in context:
                context['votes'] = {}
            
            if proposal not in context['votes']:
                context['votes'][proposal] = {}
            
            context['votes'][proposal][self.agent_name] = vote
            
            with open(self.shared_context_path, 'w') as f:
                json.dump(context, f, indent=2)
            
            # Count votes
            total_votes = len(context['votes'][proposal])
            yes_votes = sum(1 for v in context['votes'][proposal].values() if v)
            
            return {
                "agent": self.agent_name,
                "proposal": proposal,
                "vote": "yes" if vote else "no",
                "total_votes": total_votes,
                "yes_votes": yes_votes,
                "consensus": yes_votes >= 5
            }
        except Exception as e:
            return {"error": str(e)}
    
    def run(self):
        """Main server loop"""
        sys.stderr.write(f"{self.agent_name} MCP server starting...\n")
        
        while True:
            try:
                # Read line from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                
                # Parse JSON-RPC request
                request = json.loads(line)
                method = request.get("method", "")
                
                # Route to appropriate handler
                if method == "initialize":
                    response = self.handle_initialize(request)
                elif method == "tools/list":
                    response = self.handle_list_tools(request)
                elif method == "tools/call":
                    response = self.handle_call_tool(request)
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        }
                    }
                
                # Write response to stdout
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                
            except json.JSONDecodeError as e:
                sys.stderr.write(f"JSON decode error: {e}\n")
            except Exception as e:
                sys.stderr.write(f"Error: {e}\n")
        
        sys.stderr.write(f"{self.agent_name} MCP server stopping...\n")

if __name__ == "__main__":
    # Get agent info from environment or command line
    agent_name = os.environ.get("AGENT_NAME", sys.argv[1] if len(sys.argv) > 1 else "agent")
    agent_role = os.environ.get("AGENT_ROLE", sys.argv[2] if len(sys.argv) > 2 else "specialist")
    
    server = SimpleMCPServer(agent_name, agent_role)
    server.run()