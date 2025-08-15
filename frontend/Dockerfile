# Multi-stage build for React frontend
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files (Vite outputs to dist, not build)
COPY --from=builder /app/dist /usr/share/nginx/html

# Custom nginx config
RUN echo 'server { \
    listen 3000; \
    location / { \
        root /usr/share/nginx/html; \
        try_files $uri /index.html; \
    } \
    location /api { \
        proxy_pass http://bot3-config-service:8001; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]