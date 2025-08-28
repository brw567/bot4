#!/usr/bin/env python3
"""
MCP Server for morgan (Quality Gate)
Auto-generated integration server
"""

import json
import asyncio
import sys
import os
from typing import Dict, List, Any, Optional
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

class QualitygateAgent:
    """Quality Gate Agent - morgan"""
    
    def __init__(self):
        self.name = "morgan"
        self.role = "Quality Gate"
        self.shared_context_path = "/home/hamster/bot4/.mcp/shared_context.json"
        
    async def process_task(self, task: str) -> Dict[str, Any]:
        """Process a task assigned to this agent"""
        # Load shared context
        context = {}
        if os.path.exists(self.shared_context_path):
            with open(self.shared_context_path, 'r') as f:
                context = json.load(f)
        
        # Update with our analysis
        if 'agents' not in context:
            context['agents'] = {}
        
        context['agents'][self.name] = {
            'status': 'active',
            'current_task': task,
            'last_update': str(asyncio.get_event_loop().time())
        }
        
        # Save context
        with open(self.shared_context_path, 'w') as f:
            json.dump(context, f, indent=2)
        
        return {
            'status': 'success',
            'agent': self.name,
            'role': self.role,
            'task': task,
            'message': f'{self.name} is analyzing: {task}'
        }
    
    async def vote(self, proposal: str, vote: bool) -> Dict[str, Any]:
        """Vote on a proposal"""
        # Record vote in shared context
        context = {}
        if os.path.exists(self.shared_context_path):
            with open(self.shared_context_path, 'r') as f:
                context = json.load(f)
        
        if 'votes' not in context:
            context['votes'] = {}
        
        context['votes'][proposal] = context['votes'].get(proposal, {})
        context['votes'][proposal][self.name] = vote
        
        with open(self.shared_context_path, 'w') as f:
            json.dump(context, f, indent=2)
        
        # Count votes
        total_votes = len(context['votes'][proposal])
        yes_votes = sum(1 for v in context['votes'][proposal].values() if v)
        
        return {
            'status': 'success',
            'agent': self.name,
            'vote': 'yes' if vote else 'no',
            'total_votes': total_votes,
            'yes_votes': yes_votes,
            'consensus': yes_votes >= 5
        }

async def main():
    """Main MCP server loop"""
    agent = QualitygateAgent()
    server = Server("qualitygate")
    
    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="process_task",
                description=f"Assign a task to {agent.name}",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"}
                    },
                    "required": ["task"]
                }
            ),
            types.Tool(
                name="vote",
                description=f"Vote on a proposal as {agent.name}",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "proposal": {"type": "string"},
                        "vote": {"type": "boolean"}
                    },
                    "required": ["proposal", "vote"]
                }
            ),
            types.Tool(
                name="get_status",
                description=f"Get {agent.name}'s current status",
                inputSchema={"type": "object"}
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Optional[Dict] = None) -> List[types.TextContent]:
        if name == "process_task":
            result = await agent.process_task(arguments.get("task", ""))
        elif name == "vote":
            result = await agent.vote(
                arguments.get("proposal", ""),
                arguments.get("vote", False)
            )
        elif name == "get_status":
            result = {
                "status": "active",
                "agent": agent.name,
                "role": agent.role,
                "ready": True
            }
        else:
            result = {"status": "error", "message": f"Unknown tool: {name}"}
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
