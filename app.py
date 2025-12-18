import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model/brain_tumor_model.h5")

st.title("🧠 Brain Tumor Detection")
st.write("Upload a Brain MRI image to detect tumor")

uploaded_file = st.file_uploader("Choose MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.error("🛑 Tumor Detected")
    else:
        st.success("✅ No Tumor Detected")
