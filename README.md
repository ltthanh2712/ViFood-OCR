# ViFood-OCR

Hệ thống OCR cho nhận dạng thông tin dinh dưỡng trên bao bì thực phẩm Việt Nam, sử dụng CRAFT (text detection) và VietOCR (text recognition).

## Tính năng

- Phát hiện vùng text trên ảnh bao bì thực phẩm
- Nhận dạng chữ Tiếng Việt với độ chính xác cao
- Sắp xếp và nhóm text theo hàng
- Xuất kết quả OCR ra file text

## Yêu cầu hệ thống

- Python 3.11
- RAM tối thiểu: 4GB
- Disk space: ~2GB (bao gồm model weights)

## Cài đặt

### 1. Clone repository:

```bash
git clone https://github.com/ltthanh2712/ViFood-OCR.git
cd ViFood-OCR
```

### 2. Tạo Virtual Environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# hoặc: .venv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Chạy chương trình:

```bash
python main.py
```

## Sử dụng

### Cấu trúc thư mục:

```
ViFood-OCR/
├── images/           # Thư mục chứa ảnh đầu vào
├── result/           # Thư mục chứa kết quả đầu ra
├── output/           # Thư mục chứa kết quả chi tiết
├── CRAFT-pytorch/    # Module phát hiện text
│   └── weights/      # Model weights
└── VietOCR/          # Module nhận dạng text
```

### Thêm ảnh mới để xử lý:

1. Đặt ảnh vào thư mục `images/`
2. Sửa đường dẫn trong [main.py](main.py#L585):
   ```python
   image_path = "./images/your_image.jpg"
   ```
3. Chạy lại chương trình

### Kết quả:

Sau khi chạy, kết quả sẽ được lưu vào:
- `ocr_result.txt` - Kết quả chi tiết với tọa độ
- `ocr_simple.txt` - Kết quả đơn giản, chỉ text
- `debug_order.jpg` - Hình ảnh debug thứ tự đọc
- `detection_result.jpg` - Hình ảnh với bounding boxes
- `heatmap_result.jpg` - Heatmap của CRAFT
- `cropped_regions.jpg` - Các vùng text được crop

## Yêu cầu hệ thống

### Docker:
- Docker version 20.10+
- Docker Compose version 1.29+

### Local:
- Python 3.11
- CUDA (optional, cho GPU acceleration)
- RAM tối thiểu: 4GB
- Disk space: ~2GB (bao gồm model weights)

## Models

- **CRAFT**: Text detection model - `craft_mlt_25k.pth`
- **VietOCR**: Vietnamese text recognition model - Tự động tải về khi chạy lần đầu

## Troubleshooting

### Lỗi NumPy incompatibility:
```bash
pip install "numpy<2"
```

### Lỗi OpenCV:
```bash
pip uninstall opencv-python-headless -y
pip install opencv-python==4.9.0.80
```

### Container không tìm thấy ảnh:
Kiểm tra volume mounting trong `docker-compose.yml` hoặc lệnh `docker run`

## License

MITModels

- **CRAFT**: Text detection model - `craft_mlt_25k.pth` (đã có sẵn trong repo)
- **VietOCR**: Vietnamese text recognition model - Tự động tải về khi chạy lần đầu

## Troubleshooting

### Lỗi NumPy incompatibility:
```bash
pip install "numpy<2"
```

### Lỗi OpenCV:
```bash
pip uninstall opencv-python-headless -y
pip install opencv-python==4.9.0.80
``