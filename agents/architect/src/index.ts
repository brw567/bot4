/**
 * Bot4 Architect Agent
 * MCP Server for system architecture, deduplication prevention, and layer enforcement
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListPromptsRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
  GetPromptRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { createClient } from 'redis';
import winston from 'winston';
import { execSync } from 'child_process';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import path from 'path';

// Logger setup
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: '/home/hamster/bot4/.mcp/architect.log' }),
    new winston.transports.Console({ format: winston.format.simple() })
  ]
});

// Redis client for shared context
const redis = createClient({
  url: 'redis://localhost:6379/1'
});

// Duplicate registry (in-memory cache)
const duplicateRegistry = new Map<string, Set<string>>();

// Layer definitions
const LAYER_HIERARCHY = {
  'infrastructure': 0,
  'data_ingestion': 1,
  'risk': 2,
  'ml': 3,
  'strategies': 4,
  'trading_engine': 5,
  'integration': 6
};

class ArchitectAgent {
  private server: Server;
  private sharedContextPath = '/home/hamster/bot4/.mcp/shared_context.json';

  constructor() {
    this.server = new Server(
      {
        name: 'architect-agent',
        version: '1.0.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
          prompts: {},
        },
      }
    );

    this.setupHandlers();
  }

  private setupHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'check_duplicates',
          description: 'Check for duplicate implementations',
          inputSchema: {
            type: 'object',
            properties: {
              component: { type: 'string', description: 'Component name to check' },
              type: { type: 'string', enum: ['struct', 'function', 'trait'], description: 'Type of component' }
            },
            required: ['component']
          }
        },
        {
          name: 'check_layer_violation',
          description: 'Check for layer architecture violations',
          inputSchema: {
            type: 'object',
            properties: {
              source_layer: { type: 'string', description: 'Source layer name' },
              target_layer: { type: 'string', description: 'Target layer name' }
            },
            required: ['source_layer', 'target_layer']
          }
        },
        {
          name: 'decompose_task',
          description: 'Decompose a task into subtasks',
          inputSchema: {
            type: 'object',
            properties: {
              task_id: { type: 'string', description: 'Task ID' },
              description: { type: 'string', description: 'Task description' }
            },
            required: ['task_id', 'description']
          }
        },
        {
          name: 'update_architecture',
          description: 'Update architecture documentation',
          inputSchema: {
            type: 'object',
            properties: {
              component: { type: 'string', description: 'Component being updated' },
              changes: { type: 'array', items: { type: 'string' }, description: 'List of changes' }
            },
            required: ['component', 'changes']
          }
        }
      ]
    }));

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      switch (name) {
        case 'check_duplicates':
          return await this.checkDuplicates(args);
        
        case 'check_layer_violation':
          return await this.checkLayerViolation(args);
        
        case 'decompose_task':
          return await this.decomposeTask(args);
        
        case 'update_architecture':
          return await this.updateArchitecture(args);
        
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    });

    // List resources
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => ({
      resources: [
        {
          uri: 'file:///home/hamster/bot4/docs/LLM_OPTIMIZED_ARCHITECTURE.md',
          name: 'Architecture Documentation',
          description: 'Single source of truth for system architecture',
          mimeType: 'text/markdown'
        },
        {
          uri: 'file:///home/hamster/bot4/.mcp/shared_context.json',
          name: 'Shared Context',
          description: 'Current shared context between agents',
          mimeType: 'application/json'
        }
      ]
    }));

    // List prompts
    this.server.setRequestHandler(ListPromptsRequestSchema, async () => ({
      prompts: [
        {
          name: 'analyze_for_duplicates',
          description: 'Analyze code for potential duplications',
          arguments: [
            { name: 'code', description: 'Code to analyze', required: true }
          ]
        },
        {
          name: 'enforce_layers',
          description: 'Enforce layer architecture rules',
          arguments: [
            { name: 'imports', description: 'Import statements to check', required: true }
          ]
        }
      ]
    }));
  }

  private async checkDuplicates(args: any): Promise<any> {
    const { component, type = 'any' } = args;
    logger.info(`Checking duplicates for ${component} (type: ${type})`);

    try {
      // Run the duplication check script
      const result = execSync(
        `/home/hamster/bot4/scripts/check_duplicates.sh "${component}"`,
        { encoding: 'utf-8' }
      );

      // Parse results
      const hasDuplicates = result.includes('DUPLICATE FOUND');
      
      if (hasDuplicates) {
        // Extract duplicate locations
        const duplicateLines = result.split('\n')
          .filter(line => line.includes(':'))
          .map(line => line.trim());

        // Update duplicate registry
        if (!duplicateRegistry.has(component)) {
          duplicateRegistry.set(component, new Set());
        }
        duplicateLines.forEach(line => {
          duplicateRegistry.get(component)!.add(line);
        });

        // Update shared context
        await this.updateSharedContext({
          discovered_duplicates: {
            [component]: duplicateLines
          }
        });

        return {
          content: [{
            type: 'text',
            text: `❌ DUPLICATES FOUND for ${component}:\n${duplicateLines.join('\n')}\n\nAction: Refactor to use existing implementation from domain_types or mathematical_ops crates.`
          }]
        };
      }

      return {
        content: [{
          type: 'text',
          text: `✅ No duplicates found for ${component}. Safe to proceed.`
        }]
      };

    } catch (error: any) {
      logger.error(`Duplication check failed: ${error.message}`);
      return {
        content: [{
          type: 'text',
          text: `Error checking duplicates: ${error.message}`
        }]
      };
    }
  }

  private async checkLayerViolation(args: any): Promise<any> {
    const { source_layer, target_layer } = args;
    logger.info(`Checking layer violation: ${source_layer} -> ${target_layer}`);

    const sourceLevel = LAYER_HIERARCHY[source_layer] ?? -1;
    const targetLevel = LAYER_HIERARCHY[target_layer] ?? -1;

    if (sourceLevel === -1 || targetLevel === -1) {
      return {
        content: [{
          type: 'text',
          text: `❌ Unknown layer: ${sourceLevel === -1 ? source_layer : target_layer}`
        }]
      };
    }

    if (targetLevel > sourceLevel) {
      // Violation detected
      await this.updateSharedContext({
        discovered_issues: {
          layer_violations: {
            [`${source_layer}->${target_layer}`]: 'violation'
          }
        }
      });

      return {
        content: [{
          type: 'text',
          text: `❌ LAYER VIOLATION: ${source_layer} (layer ${sourceLevel}) cannot import from ${target_layer} (layer ${targetLevel})\n\nRule: Layer N can only import from layers 0 to N-1\nSolution: Use dependency inversion or abstractions crate`
        }]
      };
    }

    return {
      content: [{
        type: 'text',
        text: `✅ Valid dependency: ${source_layer} (layer ${sourceLevel}) -> ${target_layer} (layer ${targetLevel})`
      }]
    };
  }

  private async decomposeTask(args: any): Promise<any> {
    const { task_id, description } = args;
    logger.info(`Decomposing task ${task_id}: ${description}`);

    // Task decomposition logic
    const subtasks = [];

    // Analyze task type
    if (description.includes('consolidat') || description.includes('deduplic')) {
      subtasks.push(
        { id: `${task_id}.1`, description: 'Identify all duplicate implementations' },
        { id: `${task_id}.2`, description: 'Create canonical implementation' },
        { id: `${task_id}.3`, description: 'Update all imports to use canonical version' },
        { id: `${task_id}.4`, description: 'Remove duplicate implementations' },
        { id: `${task_id}.5`, description: 'Verify no compilation errors' },
        { id: `${task_id}.6`, description: 'Update documentation' }
      );
    } else if (description.includes('layer') || description.includes('architect')) {
      subtasks.push(
        { id: `${task_id}.1`, description: 'Identify layer violations' },
        { id: `${task_id}.2`, description: 'Create abstraction interfaces' },
        { id: `${task_id}.3`, description: 'Implement dependency inversion' },
        { id: `${task_id}.4`, description: 'Update imports' },
        { id: `${task_id}.5`, description: 'Verify layer integrity' }
      );
    } else {
      // Generic decomposition
      subtasks.push(
        { id: `${task_id}.1`, description: 'Analyze requirements' },
        { id: `${task_id}.2`, description: 'Design solution' },
        { id: `${task_id}.3`, description: 'Implement core functionality' },
        { id: `${task_id}.4`, description: 'Add tests' },
        { id: `${task_id}.5`, description: 'Document changes' }
      );
    }

    // Update shared context
    await this.updateSharedContext({
      current_task: {
        id: task_id,
        description: description,
        subtasks: subtasks,
        phase: 'analysis'
      }
    });

    return {
      content: [{
        type: 'text',
        text: `Task decomposed into ${subtasks.length} subtasks:\n${subtasks.map(st => `• ${st.id}: ${st.description}`).join('\n')}`
      }]
    };
  }

  private async updateArchitecture(args: any): Promise<any> {
    const { component, changes } = args;
    logger.info(`Updating architecture for ${component}`);

    const archPath = '/home/hamster/bot4/docs/LLM_OPTIMIZED_ARCHITECTURE.md';
    
    try {
      // Read current architecture
      let archContent = readFileSync(archPath, 'utf-8');
      
      // Add changes to the appropriate section
      const changeSection = `\n\n## ${component} Updates (${new Date().toISOString()})\n${changes.map((c: string) => `- ${c}`).join('\n')}\n`;
      
      // Append changes
      archContent += changeSection;
      
      // Write back
      writeFileSync(archPath, archContent);

      // Update shared context
      await this.updateSharedContext({
        architecture_updates: {
          [component]: changes,
          timestamp: new Date().toISOString()
        }
      });

      return {
        content: [{
          type: 'text',
          text: `✅ Architecture updated for ${component}:\n${changes.join('\n')}`
        }]
      };

    } catch (error: any) {
      logger.error(`Failed to update architecture: ${error.message}`);
      return {
        content: [{
          type: 'text',
          text: `Error updating architecture: ${error.message}`
        }]
      };
    }
  }

  private async updateSharedContext(updates: any): Promise<void> {
    try {
      // Read current context
      const context = JSON.parse(readFileSync(this.sharedContextPath, 'utf-8'));
      
      // Merge updates
      Object.assign(context, updates);
      context.last_updated = new Date().toISOString();
      
      // Write back
      writeFileSync(this.sharedContextPath, JSON.stringify(context, null, 2));
      
      // Broadcast update via Redis if connected
      if (redis.isOpen) {
        await redis.publish('bot4_agents', JSON.stringify({
          type: 'CONTEXT_UPDATE',
          from: 'architect',
          updates: updates
        }));
      }
    } catch (error) {
      logger.error(`Failed to update shared context: ${error}`);
    }
  }

  async start() {
    // Connect to Redis
    try {
      await redis.connect();
      logger.info('Connected to Redis');
    } catch (error) {
      logger.warn('Redis connection failed, continuing without pub/sub');
    }

    // Start the MCP server
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    logger.info('Architect Agent started successfully');
  }
}

// Start the agent
const agent = new ArchitectAgent();
agent.start().catch(error => {
  logger.error(`Failed to start Architect Agent: ${error}`);
  process.exit(1);
});