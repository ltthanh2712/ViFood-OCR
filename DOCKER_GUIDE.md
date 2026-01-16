# Docker Deployment Guide - ViFood OCR API

Hướng dẫn build và deploy ViFood OCR API bằng Docker.

## Prerequisites

- Docker Desktop đã được cài đặt và đang chạy
- Docker Compose (đi kèm với Docker Desktop)

## Build Docker Image

### Cách 1: Build với Docker Compose (Khuyến nghị)

```bash
docker-compose build
```

### Cách 2: Build trực tiếp với Docker

```bash
docker build -t vifood-ocr:latest .
```

Quá trình build sẽ:
- Cài đặt tất cả dependencies
- Tải model weights tự động
- Tạo image hoàn chỉnh (~2-3GB)

## Run Container

### Cách 1: Chạy với Docker Compose

```bash
# Chạy container
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng container
docker-compose down
```

### Cách 2: Chạy trực tiếp với Docker

```bash
docker run -d \
  --name vifood-ocr-api \
  -p 8000:8000 \
  -v ./images:/app/images \
  -v ./result:/app/result \
  vifood-ocr:latest
```

## Kiểm tra Container

```bash
# Kiểm tra container đang chạy
docker ps

# Xem logs
docker logs vifood-ocr-api

# Xem logs realtime
docker logs -f vifood-ocr-api
```

## Test API

Sau khi container chạy, truy cập:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs

Test với cURL:
```bash
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@./images/test11.jpg"
```

## Export/Import Docker Image

### Export image ra file

```bash
# Export image thành file .tar
docker save vifood-ocr:latest -o vifood-ocr.tar

# Hoặc nén lại để file nhỏ hơn
docker save vifood-ocr:latest | gzip > vifood-ocr.tar.gz
```

### Import image từ file

```bash
# Import từ .tar
docker load -i vifood-ocr.tar

# Import từ .tar.gz
docker load < vifood-ocr.tar.gz
```

## Deploy lên Server

### Bước 1: Chuyển image lên server

Có 3 cách:

**Cách 1: Sử dụng file .tar (Không cần Docker Registry)**

```bash
# Trên máy local
docker save vifood-ocr:latest | gzip > vifood-ocr.tar.gz

# Copy file lên server (sử dụng scp hoặc ftp)
scp vifood-ocr.tar.gz user@server:/path/to/destination/

# Trên server
docker load < vifood-ocr.tar.gz
```

**Cách 2: Push lên Docker Hub (Public/Private)**

```bash
# Tag image
docker tag vifood-ocr:latest your-dockerhub-username/vifood-ocr:latest

# Login Docker Hub
docker login

# Push image
docker push your-dockerhub-username/vifood-ocr:latest

# Trên server, pull image
docker pull your-dockerhub-username/vifood-ocr:latest
```

**Cách 3: Private Docker Registry**

```bash
# Tag cho private registry
docker tag vifood-ocr:latest registry.your-domain.com/vifood-ocr:latest

# Push
docker push registry.your-domain.com/vifood-ocr:latest
```

### Bước 2: Chạy trên server

Tạo file `docker-compose.yml` trên server:

```yaml
version: '3.8'

services:
  vifood-ocr:
    image: vifood-ocr:latest
    container_name: vifood-ocr-api
    ports:
      - "8000:8000"
    volumes:
      - ./images:/app/images
      - ./result:/app/result
    restart: unless-stopped
```

Chạy:
```bash
docker-compose up -d
```

## Production Tips

### 1. Sử dụng Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Tăng timeout cho OCR processing
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }
}
```

### 2. Giới hạn tài nguyên

```yaml
services:
  vifood-ocr:
    image: vifood-ocr:latest
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 3. Scale multiple containers

```bash
# Chạy 3 instances
docker-compose up -d --scale vifood-ocr=3

# Sử dụng load balancer (nginx/traefik) để phân phối request
```

### 4. Monitoring

```bash
# Xem resource usage
docker stats vifood-ocr-api

# Health check
curl http://localhost:8000/
```

## Troubleshooting

### Container không start

```bash
# Xem logs chi tiết
docker logs vifood-ocr-api

# Chạy interactive mode để debug
docker run -it --rm vifood-ocr:latest /bin/bash
```

### Out of memory

Tăng memory limit trong docker-compose.yml hoặc Docker Desktop settings.

### Port đã được sử dụng

Đổi port mapping trong docker-compose.yml:
```yaml
ports:
  - "8080:8000"  # Thay vì 8000:8000
```

## Cleanup

```bash
# Dừng và xóa container
docker-compose down

# Xóa image
docker rmi vifood-ocr:latest

# Xóa tất cả unused images
docker image prune -a
```

## Size Optimization

Để giảm size của image:

1. Sử dụng multi-stage build
2. Xóa cache pip: `--no-cache-dir`
3. Xóa apt cache: `rm -rf /var/lib/apt/lists/*`
4. Chỉ copy file cần thiết

## Summary

**Quick Start:**
```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Check
curl http://localhost:8000/

# Export
docker save vifood-ocr:latest | gzip > vifood-ocr.tar.gz

# Deploy to server
# 1. Copy vifood-ocr.tar.gz to server
# 2. docker load < vifood-ocr.tar.gz
# 3. docker-compose up -d
```

**Estimated Sizes:**
- Docker image: ~2-3GB
- Compressed .tar.gz: ~1-1.5GB
