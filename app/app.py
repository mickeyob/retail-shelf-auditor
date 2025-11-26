import gradio as gr
import cv2
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- 1. SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")

if not os.path.exists(MODEL_PATH):
    print(f"⚠️ Model not found at {MODEL_PATH}.")

print(f"📂 Loading model from: {MODEL_PATH}...")

try:
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=MODEL_PATH,
        # Keep confidence low (0.25) to catch more items
        confidence_threshold=0.25, 
        device="cpu" 
    )
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    detection_model = None

# --- 2. LOGIC ---
def process_image(image, slice_size, overlap_ratio):
    if image is None: return None, "⚠️ No image."
    if detection_model is None: return image, "❌ Model Error."
        
    image_pil = Image.fromarray(image)
    
    # SAHI Inference
    result = get_sliced_prediction(
        image_pil,
        detection_model,
        slice_height=int(slice_size),
        slice_width=int(slice_size),
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio
    )
    
    # Visualization
    output_img = image.copy()
    for prediction in result.object_prediction_list:
        bbox = prediction.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        
    return output_img, f"✅ Detected {len(result.object_prediction_list)} items."

# --- 3. UI ---
# UPDATED TITLE HERE
with gr.Blocks(title="Retail Shelf Auditor (YOLOv8 + SAHI)") as demo:
    # UPDATED HEADER HERE
    gr.Markdown("# 🏪 Retail Shelf Auditor (YOLOv8 + SAHI)")
    gr.Markdown("Upload an image to test the detection pipeline.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input", type="numpy")
            btn = gr.Button("Audit Shelf", variant="primary")
            
            # Tuning Sliders
            with gr.Accordion("Advanced Settings", open=True):
                slice_slider = gr.Slider(320, 1280, value=512, step=32, label="Slice Size (Smaller = More Detailed)")
                overlap_slider = gr.Slider(0, 0.6, value=0.35, label="Overlap Ratio")

            # Examples (Only shows if files exist)
            examples_list = []
            if os.path.exists("test1.jpg"): examples_list.append("test1.jpg")
            if os.path.exists("test2.jpg"): examples_list.append("test2.jpg")
            if os.path.exists("test3.jpg"): examples_list.append("test3.jpg")
            
            if examples_list:
                gr.Examples(examples=examples_list, inputs=input_img, label="Test Images")

        with gr.Column():
            output_img = gr.Image(label="Result")
            output_text = gr.Textbox(label="Count")

    btn.click(process_image, [input_img, slice_slider, overlap_slider], [output_img, output_text])

if __name__ == "__main__":
    demo.launch(ssr_mode=False)