# ViFood OCR API

REST API service để nhận dạng thông tin dinh dưỡng trên bao bì thực phẩm Việt Nam.

## Cài đặt

```bash
pip install fastapi uvicorn python-multipart requests
```

Hoặc cài đặt tất cả dependencies:

```bash
pip install -r requirements.txt
```

## Chạy API Server

```bash
python api.py
```

Server sẽ chạy tại: `http://localhost:8000`

API Documentation (Swagger UI): `http://localhost:8000/docs`

## API Endpoints

### 1. Health Check

**GET** `/`

Kiểm tra trạng thái API server.

**Response:**
```json
{
  "message": "ViFood OCR API is running",
  "status": "healthy",
  "version": "1.0.0"
}
```

### 2. OCR - Chi tiết

**POST** `/ocr`

Nhận ảnh và trả về kết quả OCR đầy đủ với metadata.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: File ảnh (jpg, jpeg, png, bmp)

**Response:**
```json
{
  "success": true,
  "text_regions": 88,
  "rows": 15,
  "results": [
    {
      "row_number": 1,
      "text": "THÀNH PHẦN DINH DƯỜNG TRUNG BÌNH",
      "items": [
        {"text": "THÀNH PHẦN", "confidence": 1.0},
        {"text": "DINH DƯỜNG", "confidence": 1.0}
      ]
    }
  ],
  "full_text": "THÀNH PHẦN DINH DƯỜNG TRUNG BÌNH\n..."
}
```

### 3. OCR - Đơn giản

**POST** `/ocr/simple`

Nhận ảnh và trả về chỉ text (không có metadata).

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: File ảnh

**Response:**
```json
{
  "success": true,
  "text": "THÀNH PHẦN DINH DƯỜNG TRUNG BÌNH\nNăng lượng/ Energy 94 kcal\n..."
}
```

## Sử dụng API

### Python với requests

```python
import requests

# Upload ảnh
with open('image.jpg', 'rb') as f:
    files = {'file': ('image.jpg', f, 'image/jpeg')}
    response = requests.post('http://localhost:8000/ocr', files=files)
    result = response.json()
    print(result['full_text'])
```

### cURL

```bash
# OCR chi tiết
curl -X POST "http://localhost:8000/ocr" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"

# OCR đơn giản
curl -X POST "http://localhost:8000/ocr/simple" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### JavaScript (Fetch API)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/ocr', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('OCR Result:', data.full_text);
})
.catch(error => console.error('Error:', error));
```

## Test API

Sử dụng script test có sẵn:

```bash
python test_api.py
```

Script này sẽ:
1. Test health check endpoint
2. Test OCR endpoint với ảnh mẫu
3. Test simple OCR endpoint
4. Lưu kết quả vào file `api_test_result.json`

## Cấu hình

### Thay đổi Port

Trong file `api.py`, dòng cuối cùng:

```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # Đổi port tại đây
```

### Chạy production mode

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### CORS

API đã cấu hình CORS cho phép truy cập từ mọi origin. Để giới hạn, sửa trong `api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Chỉ cho phép domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Performance

- Model được load 1 lần khi server khởi động
- CPU mode: ~3-5 giây/ảnh
- GPU mode: ~1-2 giây/ảnh (cần cấu hình CUDA)

## Error Codes

- `200`: Success
- `400`: Bad request (file không hợp lệ)
- `500`: Server error (lỗi xử lý)

## Lưu ý

- API chỉ chấp nhận file ảnh (jpg, jpeg, png, bmp)
- Kích thước file tối đa: 10MB (mặc định FastAPI)
- Server mặc định chạy ở CPU mode
- Để sử dụng GPU, sửa `config['device'] = 'cuda'` trong file api.py
