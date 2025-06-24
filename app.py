import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import tempfile
from ultralytics import YOLO

# Set page config
st.set_page_config(page_title="Marine Life Detector 🌊", layout="centered")

# Title
st.title("🐠 Marine Life Detector")
st.caption("Powered by YOLOv8m — Detects underwater animals from uploaded images")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("marine_detector.pt")

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📤 Upload an underwater image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image using PIL
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Save image to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        image_path = tmp_file.name

    # Run YOLOv8 inference
    with st.spinner("Detecting marine life... 🐋"):
        results = model.predict(image_path, conf=0.1, save=False)[0]
        boxes = results.boxes

        # Draw results on the original image using PIL
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                text = f"{label} ({conf:.2f})"

                # Draw bounding box
                draw.rectangle(xyxy, outline="red", width=3)
                draw.text((xyxy[0], xyxy[1] - 12), text, fill="white", font=font)

            st.success("✅ Detection complete!")
            st.image(image, caption="🔍 Detected Marine Life", use_column_width=True)

            # Optional: list all detections below
            st.subheader("📋 Detection Summary")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                st.markdown(f"- **{label}** — `{conf:.2f}` confidence")
        else:
            st.warning("No marine animals were detected in this image.")
