# ================================
# Lab 5: Computer Vision
# Image Classification Web App
# Using ResNet18 + Streamlit (CPU)
# ================================

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd

# -------------------------------
# Step 1: Page configuration
# -------------------------------
st.set_page_config(
    page_title="Computer Vision Image Classifier",
    layout="wide"
)

st.title("🖼️ Image Classification using ResNet18 (CPU)")
st.write("Upload an image to classify using a pre-trained ResNet18 model.")

# -------------------------------
# Step 2 & 3: Load model (CPU only)
# -------------------------------
device = torch.device("cpu")

weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()
model.to(device)

class_names = weights.meta["categories"]

# -------------------------------
# Step 5: Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------
# Step 6: Image uploader
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # Step 7: Model inference
    # -------------------------------
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)

    # -------------------------------
    # Step 8: Top-5 predictions
    # -------------------------------
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    results = []
    for i in range(5):
        results.append({
            "Class": class_names[top5_idx[0][i]],
            "Probability": float(top5_prob[0][i])
        })

    df = pd.DataFrame(results)

    with col2:
        st.subheader("🔍 Top-5 Predictions")
        st.dataframe(df, use_container_width=True)

        # -------------------------------
        # Step 9: Bar chart
        # -------------------------------
        st.subheader("📊 Prediction Probabilities")
        st.bar_chart(
            df.set_index("Class")["Probability"]
        )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("BSD3513 • Lab 5 • Computer Vision • ResNet18 (CPU)")
