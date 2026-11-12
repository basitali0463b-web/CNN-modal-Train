
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠"
)

st.title("🧠 Brain Tumor MRI Classifier")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

@st.cache_resource
def load_cnn_model():
    model_path = hf_hub_download(
        repo_id="Basit026/CNN.keras",
        filename="brain_tumor_best.keras"
    )
    return tf.keras.models.load_model(model_path)

try:
    model = load_cnn_model()
    st.success("CNN model loaded successfully!")
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("Analyze MRI Image"):
        resized = image.resize((224, 224))
        image_array = np.array(resized, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]
        index = int(np.argmax(predictions))

        st.subheader("Prediction")
        st.write(f"Category: **{CLASS_NAMES[index]}**")
        st.write(f"Confidence: **{predictions[index] * 100:.2f}%**")

        st.subheader("Class probabilities")
        for name, probability in zip(CLASS_NAMES, predictions):
            st.write(f"{name}: {probability * 100:.2f}%")
            st.progress(float(probability))

st.warning(
    "Educational project only. This model is not clinically validated "
    "and must not be used for medical diagnosis."
)
