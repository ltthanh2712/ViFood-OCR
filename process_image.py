import cv2
import torch
import numpy as np
from PIL import Image
import sys
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add paths
sys.path.append('./CRAFT-pytorch')
sys.path.append('./VietOCR')

from craft import CRAFT
import craft_utils
import imgproc
import file_utils
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# Import the ViFood_OCR class from main
from main import ViFood_OCR

def main():
    parser = argparse.ArgumentParser(description='ViFood OCR - Process food packaging images')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save output files')
    parser.add_argument('--text-threshold', type=float, default=0.7, help='Text confidence threshold')
    parser.add_argument('--link-threshold', type=float, default=0.4, help='Link confidence threshold')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize OCR system
    print("Initializing OCR system...")
    ocr_system = ViFood_OCR()
    
    # Process image
    print(f"Processing image: {args.image_path}")
    boxes, texts, rows = ocr_system.process_image(args.image_path)
    
    # Save results to specified output directory
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    result_file = os.path.join(args.output_dir, f"{base_name}_ocr_result.txt")
    simple_file = os.path.join(args.output_dir, f"{base_name}_ocr_simple.txt")
    
    print("\n=== KẾT QUẢ CUỐI CÙNG ===")
    print(f"Đã xử lý {len(boxes)} vùng text trong {len(rows)} hàng")
    print(f"Kết quả đã được lưu vào:")
    print(f"  - {result_file}")
    print(f"  - {simple_file}")

if __name__ == "__main__":
    main()
