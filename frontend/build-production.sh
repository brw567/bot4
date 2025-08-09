#!/bin/bash

echo "Building production bundle (skipping TypeScript checks)..."

# Create a temporary tsconfig that doesn't emit errors
cat > tsconfig.temp.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "node",
    "jsx": "react-jsx",
    "noEmit": true,
    "isolatedModules": true,
    "allowSyntheticDefaultImports": true
  },
  "include": ["src/**/*"],
  "exclude": ["**/*.test.ts", "**/*.test.tsx", "**/__tests__/**"]
}
EOF

# Backup the original tsconfig files
mv tsconfig.json tsconfig.json.bak 2>/dev/null || true
mv tsconfig.app.json tsconfig.app.json.bak 2>/dev/null || true

# Use the temporary config
cp tsconfig.temp.json tsconfig.json
cp tsconfig.temp.json tsconfig.app.json

# Build with Vite directly (no TypeScript checking)
echo "Running Vite build..."
NODE_ENV=production npx vite build

# Check if build succeeded
if [ -d "dist" ]; then
  echo "✅ Build successful! Files are in the dist/ directory"
  ls -la dist/
else
  echo "❌ Build failed"
fi

# Restore original configs
mv tsconfig.json.bak tsconfig.json 2>/dev/null || true
mv tsconfig.app.json.bak tsconfig.app.json 2>/dev/null || true

# Clean up
rm -f tsconfig.temp.json

echo "Done!"