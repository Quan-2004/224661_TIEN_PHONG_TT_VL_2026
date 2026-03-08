# Dockerfile – Frontend mOC-iSVM
# Stage 1: Build với Node 20-alpine
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files và install dependencies
COPY frontend/package*.json ./
RUN npm ci --silent

# Copy source code
COPY frontend/ .

# Build production bundle (Vite)
ARG VITE_API_URL=http://backend:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# -------------------------------------------------------
# Stage 2: Serve với Nginx alpine
FROM nginx:alpine

# Copy build output từ stage 1
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy cấu hình Nginx
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
