from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import os
import sys
from typing import List, Dict
import tempfile
import torch

# Add paths
sys.path.append('./CRAFT-pytorch')
sys.path.append('./VietOCR')

from craft import CRAFT
import craft_utils
import imgproc
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from collections import OrderedDict

app = FastAPI(
    title="ViFood OCR API",
    description="API để nhận dạng thông tin dinh dưỡng trên bao bì thực phẩm Việt Nam",
    version="1.0.0"
)

# CORS middleware để cho phép truy cập từ frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OCR system instance
ocr_system = None

class ViFood_OCR:
    def __init__(self):
        # Initialize CRAFT for text detection
        self.craft_net = CRAFT()
        print('Loading CRAFT weights...')
        state_dict = torch.load('./CRAFT-pytorch/weights/craft_mlt_25k.pth', map_location='cpu')

        # Nếu model được train với DataParallel, loại bỏ tiền tố "module."
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        self.craft_net.load_state_dict(new_state_dict)
        self.craft_net.eval()

        # Initialize VietOCR for text recognition
        print('Loading VietOCR model...')
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = False
        self.viet_ocr = Predictor(config)
        print('Models loaded successfully!')

    def detect_text_regions(self, image):
        """Phát hiện vùng text trong ảnh sử dụng CRAFT"""
        # Resize image
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
        )
        ratio_h = ratio_w = 1 / target_ratio

        # Preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = x.unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            y, feature = self.craft_net(x)

        # Make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(
            score_text, score_link, 0.7, 0.4, 0.4, False
        )

        # Coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

        return boxes, polys

    def sort_boxes_left_to_right_top_to_bottom(self, boxes):
        """Sắp xếp boxes theo thứ tự đọc: từ trên xuống dưới, trái sang phải"""
        sorted_boxes = sorted(boxes, key=lambda x: (int(x[0][1] / 20) * 1000 + x[0][0]))
        return sorted_boxes

    def recognize_text(self, image, boxes):
        """Nhận dạng text trong các vùng đã detect"""
        texts = []
        for box in boxes:
            # Get bounding box coordinates
            x_min = int(min(box[:, 0]))
            y_min = int(min(box[:, 1]))
            x_max = int(max(box[:, 0]))
            y_max = int(max(box[:, 1]))

            # Crop region
            cropped = image[y_min:y_max, x_min:x_max]

            if cropped.size == 0:
                texts.append("")
                continue

            # Convert to PIL Image
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

            # Recognize text
            try:
                text = self.viet_ocr.predict(cropped_pil)
                texts.append(text)
            except Exception as e:
                print(f"Error recognizing text: {e}")
                texts.append("")

        return texts

    def group_text_by_rows(self, boxes, texts, row_threshold=20):
        """Nhóm text theo hàng dựa trên tọa độ y"""
        if len(boxes) == 0:
            return []

        # Tạo danh sách các item với thông tin box và text
        items = []
        for box, text in zip(boxes, texts):
            y_center = (box[0][1] + box[2][1]) / 2
            x_min = min(box[:, 0])
            items.append({
                'y_center': y_center,
                'x_min': x_min,
                'text': text,
                'box': box
            })

        # Sắp xếp theo y_center
        items.sort(key=lambda x: x['y_center'])

        # Nhóm các items vào các hàng
        rows = []
        current_row = [items[0]]

        for i in range(1, len(items)):
            if abs(items[i]['y_center'] - current_row[0]['y_center']) < row_threshold:
                current_row.append(items[i])
            else:
                # Sắp xếp current_row theo x_min (trái sang phải)
                current_row.sort(key=lambda x: x['x_min'])
                rows.append(current_row)
                current_row = [items[i]]

        # Thêm hàng cuối cùng
        if current_row:
            current_row.sort(key=lambda x: x['x_min'])
            rows.append(current_row)

        return rows

    def process_image(self, image):
        """Xử lý ảnh và trả về kết quả OCR"""
        # Detect text regions
        boxes, polys = self.detect_text_regions(image)
        print(f"Found {len(boxes)} text regions")

        if len(boxes) == 0:
            return [], [], []

        # Sort boxes
        sorted_boxes = self.sort_boxes_left_to_right_top_to_bottom(boxes)

        # Recognize text
        texts = self.recognize_text(image, sorted_boxes)

        # Group by rows
        rows = self.group_text_by_rows(sorted_boxes, texts)

        return sorted_boxes, texts, rows


@app.on_event("startup")
async def startup_event():
    """Khởi tạo OCR system khi start API"""
    global ocr_system
    print("Initializing OCR system...")
    ocr_system = ViFood_OCR()
    print("OCR system ready!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "ViFood OCR API is running",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    """
    Nhận ảnh và trả về kết quả OCR
    
    Args:
        file: File ảnh upload (jpg, jpeg, png, bmp)
    
    Returns:
        JSON với kết quả OCR bao gồm:
        - text_regions: Số lượng vùng text phát hiện được
        - rows: Số lượng hàng text
        - results: Danh sách các hàng text với nội dung
        - full_text: Toàn bộ text ghép lại
    """
    try:
        # Kiểm tra file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File phải là ảnh (jpg, jpeg, png, bmp)")
        
        # Đọc file ảnh
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Không thể đọc ảnh. Vui lòng kiểm tra định dạng file.")
        
        # Xử lý OCR
        boxes, texts, rows = ocr_system.process_image(image)
        
        # Chuẩn bị kết quả trả về
        result_rows = []
        for idx, row in enumerate(rows, 1):
            row_text = " ".join([item['text'] for item in row if item['text']])
            result_rows.append({
                "row_number": idx,
                "text": row_text,
                "items": [{"text": item['text'], "confidence": 1.0} for item in row]
            })
        
        # Tạo full text
        full_text = "\n".join([row['text'] for row in result_rows])
        
        return JSONResponse(content={
            "success": True,
            "text_regions": len(boxes),
            "rows": len(rows),
            "results": result_rows,
            "full_text": full_text
        })
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")


@app.post("/ocr/simple")
async def ocr_image_simple(file: UploadFile = File(...)):
    """
    Nhận ảnh và trả về text đơn giản (chỉ text, không có metadata)
    
    Args:
        file: File ảnh upload
    
    Returns:
        JSON với text đã nhận dạng
    """
    try:
        # Kiểm tra file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File phải là ảnh")
        
        # Đọc file ảnh
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Không thể đọc ảnh")
        
        # Xử lý OCR
        boxes, texts, rows = ocr_system.process_image(image)
        
        # Tạo text đơn giản
        result_texts = []
        for row in rows:
            row_text = " ".join([item['text'] for item in row if item['text']])
            if row_text.strip():
                result_texts.append(row_text)
        
        return JSONResponse(content={
            "success": True,
            "text": "\n".join(result_texts)
        })
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")


if __name__ == "__main__":
    # Run server
    print("Starting ViFood OCR API server...")
    print("API Documentation: http://localhost:8000/docs")
    print("API Base URL: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
