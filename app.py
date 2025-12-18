import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection")

st.title("🧠 Brain Tumor Detection using CNN")

# Load model
model = tf.keras.models.load_model("brain_tumor_dataset.h5")

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open image and FORCE RGB
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize (same as training)
    image = image.resize((150, 150))

    # Convert to array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    confidence = prediction[0][0] * 100

    if prediction[0][0] > 0.5:
        st.error(f"🛑 Tumor Detected ({confidence:.2f}%)")
    else:
        st.success(f"✅ No Tumor Detected ({100-confidence:.2f}%)")
