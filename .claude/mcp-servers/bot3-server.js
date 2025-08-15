const { Server } = require('@modelcontextprotocol/sdk');

const server = new Server({
  name: 'bot3-trading',
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
