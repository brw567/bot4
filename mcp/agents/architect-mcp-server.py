#!/usr/bin/env python3
"""
MCP Server for Architect Agent
Provides system design, deduplication detection, and architecture enforcement
"""

import json
import asyncio
import sys
import os
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Add project root to path
sys.path.append('/home/hamster/bot4')

@dataclass
class ArchitectContext:
    """Shared context for Architect agent"""
    current_task: Optional[str] = None
    duplicates_found: int = 0
    layer_violations: List[str] = None
    design_proposals: List[Dict] = None
    
    def __post_init__(self):
        if self.layer_violations is None:
            self.layer_violations = []
        if self.design_proposals is None:
            self.design_proposals = []

class ArchitectAgent:
    """Architect Agent - System Design & Deduplication"""
    
    def __init__(self):
        self.context = ArchitectContext()
        self.shared_context_path = "/home/hamster/bot4/.mcp/shared_context.json"
        self.duplicates_script = "/home/hamster/bot4/scripts/check_duplicates.sh"
        self.layer_check_script = "/home/hamster/bot4/scripts/check_layer_violations.sh"
        
    async def load_shared_context(self):
        """Load shared context from file"""
        try:
            if os.path.exists(self.shared_context_path):
                with open(self.shared_context_path, 'r') as f:
                    data = json.load(f)
                    if 'architect' in data:
                        self.context.current_task = data['architect'].get('current_task')
                        self.context.duplicates_found = data['architect'].get('duplicates_found', 0)
        except Exception as e:
            print(f"Error loading context: {e}", file=sys.stderr)
    
    async def save_shared_context(self):
        """Save shared context to file"""
        try:
            data = {}
            if os.path.exists(self.shared_context_path):
                with open(self.shared_context_path, 'r') as f:
                    data = json.load(f)
            
            data['architect'] = {
                'current_task': self.context.current_task,
                'duplicates_found': self.context.duplicates_found,
                'layer_violations': self.context.layer_violations,
                'design_proposals': self.context.design_proposals,
                'last_update': str(asyncio.get_event_loop().time())
            }
            
            with open(self.shared_context_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving context: {e}", file=sys.stderr)
    
    async def check_duplicates(self, component: str = "all") -> Dict[str, Any]:
        """Check for code duplicates"""
        try:
            result = subprocess.run(
                [self.duplicates_script, component],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse output for duplicate count
            output = result.stdout
            duplicates = []
            
            for line in output.split('\n'):
                if 'DUPLICATE FOUND' in line:
                    duplicates.append(line.strip())
            
            self.context.duplicates_found = len(duplicates)
            await self.save_shared_context()
            
            return {
                'status': 'success',
                'duplicates_found': len(duplicates),
                'details': duplicates[:10],  # First 10 duplicates
                'recommendation': 'Run deduplication sprint' if duplicates else 'No duplicates found'
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def check_layer_violations(self) -> Dict[str, Any]:
        """Check for architecture layer violations"""
        try:
            if os.path.exists(self.layer_check_script):
                result = subprocess.run(
                    [self.layer_check_script],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                violations = []
                for line in result.stdout.split('\n'):
                    if 'VIOLATION' in line:
                        violations.append(line.strip())
                
                self.context.layer_violations = violations
                await self.save_shared_context()
                
                return {
                    'status': 'success',
                    'violations_found': len(violations),
                    'details': violations[:10],
                    'recommendation': 'Fix layer boundaries' if violations else 'Architecture clean'
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'Layer check script not found',
                    'recommendation': 'Create layer validation script'
                }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def propose_design(self, feature: str, description: str) -> Dict[str, Any]:
        """Propose a system design for a feature"""
        proposal = {
            'feature': feature,
            'description': description,
            'timestamp': str(asyncio.get_event_loop().time()),
            'components': [],
            'interfaces': [],
            'dependencies': []
        }
        
        # Analyze feature and suggest design
        if 'dedup' in feature.lower():
            proposal['components'] = [
                'domain_types crate for canonical types',
                'ast-grep based duplicate detector',
                'compile-time enforcement macros'
            ]
            proposal['interfaces'] = [
                'NoDuplication trait',
                'CanonicalType trait'
            ]
        
        self.context.design_proposals.append(proposal)
        await self.save_shared_context()
        
        # Broadcast to other agents
        await self.broadcast_to_agents({
            'type': 'DESIGN_PROPOSAL',
            'from': 'architect',
            'proposal': proposal
        })
        
        return {
            'status': 'success',
            'proposal': proposal,
            'next_steps': 'Await team consensus (5/9 votes required)'
        }
    
    async def broadcast_to_agents(self, message: Dict):
        """Broadcast message to all agents via shared context"""
        try:
            data = {}
            if os.path.exists(self.shared_context_path):
                with open(self.shared_context_path, 'r') as f:
                    data = json.load(f)
            
            if 'messages' not in data:
                data['messages'] = []
            
            data['messages'].append(message)
            
            # Keep only last 100 messages
            data['messages'] = data['messages'][-100:]
            
            with open(self.shared_context_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error broadcasting: {e}", file=sys.stderr)

async def main():
    """Main MCP server loop"""
    agent = ArchitectAgent()
    await agent.load_shared_context()
    
    # Create MCP server
    server = Server("architect")
    
    # Register tools
    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="check_duplicates",
                description="Check for code duplicates in the codebase",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "component": {
                            "type": "string",
                            "description": "Component to check (default: all)"
                        }
                    }
                }
            ),
            types.Tool(
                name="check_layer_violations",
                description="Check for architecture layer violations",
                inputSchema={"type": "object"}
            ),
            types.Tool(
                name="propose_design",
                description="Propose a system design for a feature",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "feature": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["feature", "description"]
                }
            ),
            types.Tool(
                name="get_architecture_status",
                description="Get current architecture health status",
                inputSchema={"type": "object"}
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Optional[Dict] = None) -> List[types.TextContent]:
        if name == "check_duplicates":
            component = arguments.get("component", "all") if arguments else "all"
            result = await agent.check_duplicates(component)
        elif name == "check_layer_violations":
            result = await agent.check_layer_violations()
        elif name == "propose_design":
            if not arguments or 'feature' not in arguments:
                result = {"status": "error", "message": "Feature name required"}
            else:
                result = await agent.propose_design(
                    arguments['feature'],
                    arguments.get('description', '')
                )
        elif name == "get_architecture_status":
            # Combine duplicate and layer checks
            dup_result = await agent.check_duplicates()
            layer_result = await agent.check_layer_violations()
            result = {
                "status": "success",
                "duplicates": dup_result.get('duplicates_found', 0),
                "layer_violations": layer_result.get('violations_found', 0),
                "health": "critical" if dup_result.get('duplicates_found', 0) > 100 else "warning" if dup_result.get('duplicates_found', 0) > 0 else "good"
            }
        else:
            result = {"status": "error", "message": f"Unknown tool: {name}"}
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())