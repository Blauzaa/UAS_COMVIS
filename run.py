# ===================================================================
# APLIKASI UI DETEKSI ROTI DENGAN GRADIO (VERSI 2.2 - FIX TEMA & READABILITY)
# ===================================================================
import gradio as gr
import torch
import torchvision
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd

from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

print("Semua library berhasil di-import.")

# ===================================================================
# --- 1. PUSAT KENDALI & KONFIGURASI ---
# ===================================================================
MODEL_PATH = 'outputs/ssdmobilenetv3_statis_baseline_albuminatinon_test_e70_bs8_lr0_001/best_model.pth' 

CLASSES_TO_USE = [
    'baguette', 'cornbread', 'croissant', 'ensaymada', 'flatbread',
    'sourdough', 'wheat-bread', 'white-bread', 'whole-grain-bread', 'pandesal'
]
NUM_CLASSES = len(CLASSES_TO_USE)
category_map = {i + 1: name for i, name in enumerate(CLASSES_TO_USE)}
np.random.seed(42)
COLORS = np.random.randint(80, 255, size=(NUM_CLASSES, 3), dtype="uint8")


# ===================================================================
# --- 2. FUNGSI UNTUK MEMUAT MODEL ---
# ===================================================================
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan di: {model_path}\nPastikan variabel MODEL_PATH sudah benar.")
    print("Membuat kerangka model...")
    model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
    num_anchors = model.anchor_generator.num_anchors_per_location()
    in_channels = [m[0][0].in_channels for m in model.head.classification_head.module_list]
    new_head = SSDLiteClassificationHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=(NUM_CLASSES + 1), norm_layer=torch.nn.BatchNorm2d)
    model.head.classification_head = new_head
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Memuat bobot model dari: {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model berhasil dimuat dan siap digunakan!")
    return model, device

try:
    model, device = load_model(MODEL_PATH)
except FileNotFoundError as e:
    print(e)
    model, device = None, None

# ===================================================================
# --- 3. FUNGSI PREDIKSI ---
# ===================================================================
def predict_image(input_image, score_threshold):
    if input_image is None:
        return None, pd.DataFrame(columns=["Jenis Roti", "Keyakinan (%)"])

    if model is None:
        error_img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(error_img, "Model Gagal Dimuat. Periksa Path.", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2) # Warna error diubah jadi merah
        return error_img, pd.DataFrame({"Error": ["Model tidak dapat dimuat."]})

    img_pil = Image.fromarray(input_image).convert("RGB")
    img_tensor = torchvision.transforms.functional.to_tensor(img_pil)
    
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])[0]
        
    image_with_boxes = np.array(img_pil).copy()
    detection_results = []
    
    for i in range(len(prediction['scores'])):
        score = prediction['scores'][i].item()
        
        if score > score_threshold:
            box = [int(coord) for coord in prediction['boxes'][i].tolist()]
            label_id = prediction['labels'][i].item()
            class_name = category_map.get(label_id, 'N/A')
            color = [int(c) for c in COLORS[label_id - 1]]
            
            detection_results.append({
                "Jenis Roti": class_name.title(),
                "Keyakinan (%)": f"{score * 100:.1f}%"
            })
            
            label_text = f"{class_name}: {score:.2f}"
            cv2.rectangle(image_with_boxes, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(image_with_boxes, label_text, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
    summary_df = pd.DataFrame(detection_results)
    
    return image_with_boxes, summary_df

# ===================================================================
# --- 4. BUAT UI GRADIO DENGAN TEMA YANG LEBIH BAIK ---
# ===================================================================
# <-- PERBAIKAN DI SINI: Mengganti tema 'Glass' menjadi 'Soft'
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>🍞 Aplikasi Deteksi Jenis Roti v2.2</h1>
            <p>Unggah gambar roti untuk dideteksi secara <i>real-time</i>. Geser slider untuk mengatur tingkat keyakinan.</p>
        </div>
        """
    )

    with gr.Row(variant="panel"):
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 🖼️ Panel Input")
            image_input = gr.Image(type="numpy", label="Unggah Gambar Anda di Sini")
            threshold_slider = gr.Slider(
                minimum=0.1, 
                maximum=0.95, 
                value=0.5, 
                step=0.05, 
                label="Confidence Threshold",
                info="Hanya tampilkan deteksi dengan keyakinan di atas nilai ini."
            )
            
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📊 Panel Output")
            image_output = gr.Image(type="numpy", label="Hasil Deteksi")
            summary_output = gr.Dataframe(
                label="Rekap Hasil Deteksi",
                headers=["Jenis Roti", "Keyakinan (%)"],
                datatype=["str", "str"],
                row_count=(5, "dynamic"),
                col_count=(2, "fixed")
            )

    gr.Markdown(f"<p style='text-align:center; color:grey;'>Model yang digunakan: {os.path.basename(MODEL_PATH)}</p>")
    
    inputs = [image_input, threshold_slider]
    outputs = [image_output, summary_output]

    image_input.upload(fn=predict_image, inputs=inputs, outputs=outputs)
    image_input.clear(lambda: (None, None), outputs=outputs)
    threshold_slider.change(fn=predict_image, inputs=inputs, outputs=outputs)

# --- Jalankan Aplikasi ---
if __name__ == "__main__":
    print("\nMeluncurkan antarmuka Gradio v2.2...")
    print("Buka URL berikut di browser Anda:")
    demo.launch()