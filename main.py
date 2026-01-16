import cv2
import torch
import numpy as np
from PIL import Image
import sys
import os
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

class ViFood_OCR:
    def __init__(self):
        # Initialize CRAFT for text detection
        self.craft_net = CRAFT()
        print('Loading CRAFT weights...')
        state_dict = torch.load('./CRAFT-pytorch/weights/craft_mlt_25k.pth', map_location='cpu')

        # Nếu model được train với DataParallel, loại bỏ tiền tố "module."
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # bỏ "module."
            new_state_dict[name] = v

        self.craft_net.load_state_dict(new_state_dict)
        self.craft_net.eval()
        
        # Initialize VietOCR for text recognition
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained'] = False
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.vietocr_predictor = Predictor(config)
    
    def sort_boxes_left_to_right(self, boxes, tolerance=30):
        """
        Sắp xếp boxes từ trái sang phải, từ trên xuống dưới
        tolerance: cho phép sai lệch theo trục y để coi là cùng hàng
        """
        # Kiểm tra nếu boxes rỗng hoặc None
        if boxes is None or len(boxes) == 0:
            return boxes
        
        # Tính toán bounding rectangle cho mỗi box
        box_info = []
        for i, box in enumerate(boxes):
            if box is not None and len(box) > 0:
                pts = np.array(box, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                # Lưu thông tin: (x, y, w, h, box)
                box_info.append((x, y, w, h, box))
        
        if len(box_info) == 0:
            return []
        
        # Sắp xếp theo nguyên tắc reading order
        def reading_order_key(item):
            x, y, w, h, box = item
            # Sắp xếp theo y trước (hàng), sau đó theo x (cột)
            # Chia y thành các "nhóm hàng" dựa trên tolerance
            row_group = y // tolerance
            return (row_group, x)
        
        # Sắp xếp boxes theo thứ tự đọc
        sorted_box_info = sorted(box_info, key=reading_order_key)
        
        # Tinh chỉnh thêm để đảm bảo các boxes cùng hàng thực sự được nhóm lại
        final_sorted_boxes = []
        processed = set()
        
        for i, (x1, y1, w1, h1, box1) in enumerate(sorted_box_info):
            if i in processed:
                continue
            
            # Tìm tất cả boxes cùng hàng với box hiện tại
            same_row_boxes = [(x1, y1, w1, h1, box1)]
            processed.add(i)
            
            for j, (x2, y2, w2, h2, box2) in enumerate(sorted_box_info):
                if j in processed or j <= i:
                    continue
                
                # Kiểm tra overlap theo trục Y để xác định cùng hàng
                y1_center = y1 + h1 / 2
                y2_center = y2 + h2 / 2
                
                # Hai box cùng hàng nếu:
                # 1. Khoảng cách center y nhỏ hơn tolerance
                # 2. Hoặc có overlap đáng kể theo trục Y
                vertical_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                min_height = min(h1, h2)
                
                if (abs(y1_center - y2_center) <= tolerance or 
                    vertical_overlap > min_height * 0.3):
                    same_row_boxes.append((x2, y2, w2, h2, box2))
                    processed.add(j)
        
            # Sắp xếp các boxes cùng hàng theo x (từ trái sang phải)
            same_row_boxes.sort(key=lambda item: item[0])  # Sort by x coordinate
        
            # Thêm vào kết quả final
            for box_info in same_row_boxes:
                final_sorted_boxes.append(box_info[4])  # box_info[4] là box gốc
    
        return final_sorted_boxes
    
    def detect_text_regions(self, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
        """Detect text regions using CRAFT"""
        # Preprocess image
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
        )
        
        # Normalize
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            y, _ = self.craft_net(x)
        
        # Post-processing
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()
        
        # Get detection boxes
        boxes, polys = craft_utils.getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text
        )
        
        # Adjust coordinates
        ratio_h = ratio_w = 1 / target_ratio
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        
        # Sắp xếp boxes từ trái sang phải, từ trên xuống dưới
        sorted_boxes = self.sort_boxes_left_to_right(boxes)
        
        return sorted_boxes, score_text, score_link
    
    def visualize_detection(self, image, boxes, save_path="detection_result.jpg"):
        """Visualize detected text regions"""
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            vis_image = image.copy()
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(vis_image)
        
        # Kiểm tra nếu có boxes để vẽ
        if boxes is not None and len(boxes) > 0:
            # Draw bounding boxes
            for i, box in enumerate(boxes):
                if box is not None and len(box) > 0:
                    # Convert box to polygon
                    poly = np.array(box).reshape((-1, 2))
                    
                    # Create polygon patch
                    polygon = patches.Polygon(poly, linewidth=2, edgecolor='red', 
                                            facecolor='none', alpha=0.7)
                    ax.add_patch(polygon)
                    
                    # Add text label with reading order
                    center_x = np.mean(poly[:, 0])
                    center_y = np.mean(poly[:, 1])
                    ax.text(center_x, center_y, str(i+1), 
                           fontsize=12, color='yellow', weight='bold',
                           ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
                    
                    # Add arrow to show reading order
                    if i < len(boxes) - 1:
                        next_box = boxes[i + 1]
                        if next_box is not None and len(next_box) > 0:
                            next_poly = np.array(next_box).reshape((-1, 2))
                            next_center_x = np.mean(next_poly[:, 0])
                            next_center_y = np.mean(next_poly[:, 1])
                            
                            # Draw arrow
                            ax.annotate('', xy=(next_center_x, next_center_y), 
                                       xytext=(center_x, center_y),
                                       arrowprops=dict(arrowstyle='->', color='blue', 
                                                     alpha=0.6, lw=1.5))
    
        # Hiển thị số lượng boxes tìm được
        num_boxes = len(boxes) if boxes is not None else 0
        ax.set_title(f'CRAFT Text Detection - Found {num_boxes} regions (Reading Order)', fontsize=14)
        ax.axis('off')
        
        # Save image
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Đã lưu ảnh kết quả tại: {save_path}")
        
    def visualize_heatmap(self, image, score_text, score_link, save_path="heatmap_result.jpg"):
        """Visualize CRAFT heatmaps"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        if len(image.shape) == 3 and image.shape[2] == 3:
            vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            vis_image = image.copy()
        
        axes[0].imshow(vis_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Text heatmap
        axes[1].imshow(score_text, cmap='jet')
        axes[1].set_title('Text Heatmap')
        axes[1].axis('off')
        
        # Link heatmap
        axes[2].imshow(score_link, cmap='jet')
        axes[2].set_title('Link Heatmap')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Đã lưu heatmap tại: {save_path}")
    
    def crop_text_regions(self, image, boxes):
        """Crop text regions from image"""
        cropped_images = []
        
        if boxes is None or len(boxes) == 0:
            return cropped_images
        
        for box in boxes:
            if box is not None and len(box) > 0:
                # Convert to integer coordinates
                pts = np.array(box, dtype=np.int32)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(pts)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                # Crop the region
                if w > 0 and h > 0:
                    cropped = image[y:y+h, x:x+w]
                    if cropped.size > 0:
                        cropped_images.append(Image.fromarray(cropped))
                
        return cropped_images
    
    def visualize_cropped_regions(self, cropped_images, texts, save_path="cropped_regions.jpg"):
        """Visualize cropped text regions"""
        if not cropped_images:
            print("Không có vùng chữ nào để hiển thị")
            return
        
        # Calculate grid size
        n_images = len(cropped_images)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (img, text) in enumerate(zip(cropped_images, texts)):
            if i < len(axes):
                axes[i].imshow(img)
                axes[i].set_title(f'Vùng {i+1}: {text}', fontsize=10, wrap=True)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(cropped_images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Đã lưu vùng chữ cắt ra tại: {save_path}")
    
    def recognize_text(self, cropped_images):
        """Recognize text in cropped regions"""
        texts = []
        for img in cropped_images:
            try:
                text = self.vietocr_predictor.predict(img)
                texts.append(text)
            except Exception as e:
                print(f"Error recognizing text: {e}")
                texts.append("")
        return texts
    
    def debug_box_order(self, image, boxes, save_path="debug_order.jpg"):
        """Debug function to show the reading order clearly"""
        if boxes is None or len(boxes) == 0:
            print("Không có boxes để debug")
            return
            
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            vis_image = image.copy()
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(15, 10))
        ax.imshow(vis_image)
        
        # Colors for different rows
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Group boxes by rows for coloring
        box_info = []
        for i, box in enumerate(boxes):
            if box is not None and len(box) > 0:
                pts = np.array(box, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                box_info.append((i, x, y, w, h, box))
        
        # Assign row colors
        current_row = 0
        last_y = -1
        tolerance = 30
        
        for i, (idx, x, y, w, h, box) in enumerate(box_info):
            # Check if this is a new row
            if last_y != -1 and abs(y - last_y) > tolerance:
                current_row += 1
            
            poly = np.array(box).reshape((-1, 2))
            color = colors[current_row % len(colors)]
            
            # Draw polygon
            polygon = patches.Polygon(poly, linewidth=3, edgecolor=color, 
                                    facecolor='none', alpha=0.8)
            ax.add_patch(polygon)
            
            # Add large, clear text label
            center_x = np.mean(poly[:, 0])
            center_y = np.mean(poly[:, 1])
            ax.text(center_x, center_y, str(i+1), 
                   fontsize=16, color='white', weight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.9))
            
            # Add coordinates info
            ax.text(center_x, center_y - 20, f'({x},{y})', 
                   fontsize=8, color='white', weight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            
            last_y = y
        
        ax.set_title(f'Reading Order Debug - {len(boxes)} boxes', fontsize=16)
        ax.axis('off')
        
        # Add legend
        legend_text = "Thứ tự đọc: 1→2→3... (cùng màu = cùng hàng)"
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Đã lưu debug image tại: {save_path}")
    
    def group_boxes_by_rows(self, boxes, texts, tolerance=30):
        """
        Nhóm các boxes và texts theo hàng
        """
        if not boxes or not texts:
            return []
        
        # Tạo danh sách thông tin box với text tương ứng
        box_text_info = []
        for i, (box, text) in enumerate(zip(boxes, texts)):
            if box is not None and len(box) > 0:
                pts = np.array(box, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                box_text_info.append((x, y, w, h, text, i))
        
        if not box_text_info:
            return []
        
        # Sắp xếp theo y trước để dễ nhóm hàng
        box_text_info.sort(key=lambda x: x[1])  # Sort by y coordinate
        
        # Nhóm theo hàng
        rows = []
        current_row = []
        
        for i, (x, y, w, h, text, idx) in enumerate(box_text_info):
            if not current_row:
                # Hàng đầu tiên
                current_row.append((x, y, w, h, text, idx))
            else:
                # Kiểm tra xem có cùng hàng với item cuối cùng không
                last_x, last_y, last_w, last_h, last_text, last_idx = current_row[-1]
                
                # Tính overlap theo trục Y
                y_overlap = max(0, min(y + h, last_y + last_h) - max(y, last_y))
                min_height = min(h, last_h)
                
                # Cùng hàng nếu:
                # 1. Khoảng cách y nhỏ hơn tolerance
                # 2. Hoặc có overlap đáng kể theo trục Y
                if abs(y - last_y) <= tolerance or y_overlap > min_height * 0.3:
                    current_row.append((x, y, w, h, text, idx))
                else:
                    # Kết thúc hàng hiện tại, bắt đầu hàng mới
                    # Sắp xếp hàng hiện tại theo x (trái sang phải)
                    current_row.sort(key=lambda x: x[0])
                    rows.append(current_row)
                    current_row = [(x, y, w, h, text, idx)]
        
        # Thêm hàng cuối cùng
        if current_row:
            current_row.sort(key=lambda x: x[0])
            rows.append(current_row)
        
        return rows

    def save_to_txt(self, rows, save_path="ocr_result.txt"):
        """
        Lưu kết quả OCR ra file txt với format từng hàng
        """
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=== KẾT QUẢ OCR ===\n")
                f.write(f"Tổng số hàng: {len(rows)}\n")
                f.write("=" * 50 + "\n\n")
                
                for row_idx, row in enumerate(rows):
                    # Ghi thông tin hàng
                    f.write(f"Hàng {row_idx + 1}: ")
                    
                    # Ghi các text trong hàng, cách nhau bằng khoảng trắng
                    row_texts = []
                    for x, y, w, h, text, idx in row:
                        if text.strip():  # Chỉ thêm text không rỗng
                            row_texts.append(text.strip())
                    
                    # Nối các text trong hàng
                    if row_texts:
                        f.write(" ".join(row_texts))
                    else:
                        f.write("[Không có text]")
                    
                    f.write("\n")
                    
                    # Ghi chi tiết từng vùng (optional)
                    f.write(f"  Chi tiết: ")
                    for x, y, w, h, text, idx in row:
                        f.write(f"[Vùng {idx+1}: '{text.strip()}' tại ({x},{y})] ")
                    f.write("\n\n")
                
                # Thống kê
                total_regions = sum(len(row) for row in rows)
                f.write("=" * 50 + "\n")
                f.write("THỐNG KÊ:\n")
                f.write(f"- Tổng số hàng: {len(rows)}\n")
                f.write(f"- Tổng số vùng text: {total_regions}\n")
                
            print(f"Đã lưu kết quả OCR ra file: {save_path}")
            return True
            
        except Exception as e:
            print(f"Lỗi khi lưu file txt: {e}")
            return False

    def save_simple_txt(self, rows, save_path="ocr_simple.txt"):
        """
        Lưu kết quả OCR ra file txt đơn giản (chỉ text)
        """
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                for row_idx, row in enumerate(rows):
                    # Lấy text từ các vùng trong hàng
                    row_texts = []
                    for x, y, w, h, text, idx in row:
                        if text.strip():  # Chỉ thêm text không rỗng
                            row_texts.append(text.strip())
                    
                    # Ghi hàng
                    if row_texts:
                        f.write(" ".join(row_texts) + "\n")
                    else:
                        f.write("\n")  # Hàng trống
            
            print(f"Đã lưu kết quả OCR đơn giản ra file: {save_path}")
            return True
            
        except Exception as e:
            print(f"Lỗi khi lưu file txt đơn giản: {e}")
            return False

    def print_results_by_rows(self, rows):
        """
        In kết quả OCR theo từng hàng ra console
        """
        print("\n=== KẾT QUẢ OCR THEO TỪNG HÀNG ===")
        
        for row_idx, row in enumerate(rows):
            # Lấy text từ các vùng trong hàng
            row_texts = []
            for x, y, w, h, text, idx in row:
                if text.strip():
                    row_texts.append(text.strip())
            
            # In hàng
            if row_texts:
                print(f"Hàng {row_idx + 1}: {' '.join(row_texts)}")
            else:
                print(f"Hàng {row_idx + 1}: [Trống]")
        
        print(f"\nTổng số hàng: {len(rows)}")

    def process_image(self, image_path):
        """Complete OCR pipeline with visualization"""
        # Load image
        image = imgproc.loadImage(image_path)
        
        print("1. Detecting text regions...")
        boxes, score_text, score_link = self.detect_text_regions(image)
        
        print(f"2. Found {len(boxes)} text regions (sorted left-to-right, top-to-bottom)")
        
        # Debug reading order
        print("3. Debug reading order...")
        self.debug_box_order(image, boxes, "debug_order.jpg")
        
        # Visualize detection results
        print("4. Visualizing detection results...")
        self.visualize_detection(image, boxes, "detection_result.jpg")
        self.visualize_heatmap(image, score_text, score_link, "heatmap_result.jpg")
        
        # Crop regions
        cropped_images = self.crop_text_regions(image, boxes)
        
        print("5. Recognizing text...")
        texts = self.recognize_text(cropped_images)
        
        # Visualize cropped regions with recognized text
        print("6. Visualizing cropped regions...")
        self.visualize_cropped_regions(cropped_images, texts, "cropped_regions.jpg")
        
        # Group by rows và xuất kết quả
        print("7. Grouping results by rows...")
        rows = self.group_boxes_by_rows(boxes, texts)
        
        # In kết quả theo hàng
        self.print_results_by_rows(rows)
        
        # Lưu ra file txt
        print("8. Saving results to files...")
        self.save_to_txt(rows, "ocr_result.txt")
        self.save_simple_txt(rows, "ocr_simple.txt")
        
        return boxes, texts, rows

def main():
    # Initialize OCR system
    ocr_system = ViFood_OCR()
    
    # Process image
    image_path = "./images/test11.jpg"
    
    if os.path.exists(image_path):
        boxes, texts, rows = ocr_system.process_image(image_path)
        
        print("\n=== KẾT QUẢ CUỐI CÙNG ===")
        print(f"Đã xử lý {len(boxes)} vùng text trong {len(rows)} hàng")
        print("Kết quả đã được lưu vào:")
        print("  - ocr_result.txt (chi tiết)")
        print("  - ocr_simple.txt (đơn giản)")
        
    else:
        print(f"Không tìm thấy file: {image_path}")
        # List available image files
        print("Các file ảnh có sẵn:")
        for file in os.listdir("."):
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                print(f"  - {file}")

if __name__ == "__main__":
    main()