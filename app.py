import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection")

model = tf.keras.models.load_model("brain_tumor_dataset.h5")

st.title("🧠 Brain Tumor Detection using CNN")

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ⚠️ MUST MATCH TRAINING SIZE
    image = image.resize((150, 150))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    confidence = prediction[0][0] * 100

    if prediction[0][0] > 0.5:
        st.error(f"🛑 Tumor Detected ({confidence:.2f}%)")
    else:
        st.success(f"✅ No Tumor Detected ({100-confidence:.2f}%)")
