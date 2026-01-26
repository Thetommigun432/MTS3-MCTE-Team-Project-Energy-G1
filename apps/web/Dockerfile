# Stage 1: Build
FROM node:22-alpine AS builder
WORKDIR /app

# Copy workspace files from root
COPY package.json package-lock.json ./
COPY apps/web/package.json ./apps/web/

# Install dependencies deterministically
RUN npm ci --workspace=apps/web

# Copy source code
COPY apps/web ./apps/web

# Build frontend
RUN npm run build:web

# Stage 2: Serve
FROM nginx:alpine
COPY --from=builder /app/apps/web/dist /usr/share/nginx/html
COPY apps/web/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
