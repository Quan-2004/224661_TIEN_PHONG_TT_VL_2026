# Hướng dẫn Triển khai mOC-iSVM (DEPLOYMENT GUIDE)

## Phương án 1: Development Local (Windows/Mac/Linux)

### Bước 1: Cài đặt Python 3.10 và môi trường ảo

```bash
# Kiểm tra phiên bản Python
python --version   # phải là 3.10.x

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate
# Kích hoạt (Linux/Mac)
source venv/bin/activate
```

### Bước 2: Cài đặt dependencies Backend

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt

# Xuất requirements (nếu cần cập nhật)
pip freeze > backend/requirements.txt
```

### Bước 3: Cấu hình môi trường

```bash
# Chỉnh sửa backend/.env
# Thay đổi JWT_SECRET thành giá trị bảo mật!
notepad backend/.env   # Windows
nano backend/.env      # Linux/Mac
```

### Bước 4: Khởi động Backend

```bash
# Từ thư mục gốc mOC-isvm2/
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Truy cập:
# API Docs: http://localhost:8000/docs
# Health:   http://localhost:8000/health
```

### Bước 5: Khởi động Frontend

```bash
cd frontend
npm install
npm run dev

# Truy cập: http://localhost:5173
```

---

## Phương án 2: Docker Compose (Khuyến nghị cho Production)

### Yêu cầu

- Docker Engine 24.x+
- Docker Compose 2.x+

### Bước 1: Build và chạy

```bash
# Từ thư mục gốc mOC-isvm2/
docker-compose up --build -d

# Xem logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Dừng
docker-compose down
```

### Bước 2: Kiểm tra

```bash
# Backend API
curl http://localhost:8000/health

# Frontend
# Mở trình duyệt: http://localhost:3000
```

---

## Phương án 3: Deploy lên Ubuntu VPS

### Bước 1: Chuẩn bị server

```bash
# SSH vào VPS
ssh user@your-server-ip

# Cài Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Cài Docker Compose
sudo apt-get install docker-compose-plugin -y
```

### Bước 2: Upload code lên server

```bash
# Từ máy local
scp -r mOC-isvm2/ user@your-server-ip:/opt/mocsvm/

# Hoặc dùng git
git init && git add . && git commit -m "initial"
git remote add origin <your-git-repo>
git push origin main

# Trên server
cd /opt/mocsvm
git clone <your-git-repo> .
```

### Bước 3: Cấu hình Production

```bash
# Chỉnh sửa .env
nano backend/.env
# Thay JWT_SECRET bằng chuỗi ngẫu nhiên dài!
# openssl rand -hex 32

# Cập nhật CORS_ORIGINS với domain thực
# CORS_ORIGINS=https://your-domain.com
```

### Bước 4: Build và chạy Production

```bash
cd /opt/mocsvm
docker compose up --build -d

# Kiểm tra
docker compose ps
docker compose logs backend
```

### Bước 5: Cài Nginx Reverse Proxy (tùy chọn)

```bash
sudo apt-get install nginx -y

# Tạo config /etc/nginx/sites-available/mocsvm
sudo tee /etc/nginx/sites-available/mocsvm << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/mocsvm /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Bước 6: SSL với Certbot (HTTPS)

```bash
sudo apt-get install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

---

## Cấu trúc dữ liệu CSV đầu vào

### samples.csv (không header)

```
1.2,0.5,3.1,2.0
1.8,0.3,2.9,1.7
...
```

### features.csv (1 hàng)

```
feature_1,feature_2,feature_3,feature_4
```

### classes.csv (1 cột, không header)

```
successful
successful
failed
...
```

---

## Kiểm tra nhanh sau khi deploy

```bash
# 1. Đăng ký user
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123","role":"admin"}'

# 2. Đăng nhập lấy token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -d "username=admin&password=admin123" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 3. Xem danh sách models
curl http://localhost:8000/models

# 4. Test predict (cần có model được train trước)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data":[[1.2, 0.5, 3.1, 2.0]],"return_scores":true}'
```

---

## Troubleshooting

| Lỗi                           | Giải pháp                                |
| ----------------------------- | ---------------------------------------- |
| `ModuleNotFoundError: mocsvm` | Chạy uvicorn từ thư mục gốc `mOC-isvm2/` |
| Port 8000 đã dùng             | `lsof -i :8000` và kill process          |
| CORS Error                    | Kiểm tra `CORS_ORIGINS` trong `.env`     |
| JWT Invalid                   | Kiểm tra `JWT_SECRET` khớp giữa .env     |
| SQLite locked                 | Chỉ chạy một instance backend            |
