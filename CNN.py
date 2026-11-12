import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Classifier")
st.write(
    "Upload an MRI image to see which of the four dataset categories "
    "the trained CNN model predicts."
)

# Keep this file and brain_tumor_best.keras in the same folder.
MODEL_PATH = "brain_tumor_best.keras"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)


# ----------------------------
# Load trained CNN model
# ----------------------------
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_cnn_model()
    st.success("CNN model loaded successfully.")
except Exception as error:
    st.error(
        "The model could not be loaded. Make sure "
        "'brain_tumor_best.keras' is in the same folder as CNN.py."
    )
    st.exception(error)
    st.stop()


# ----------------------------
# Upload MRI image
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)

        if st.button("Analyze MRI Image", type="primary"):
            # Resize image to the model's expected input size.
            image_resized = image.resize(IMAGE_SIZE)

            # Convert image to an array and add batch dimension.
            # The trained model already contains its Rescaling layer.
            image_array = np.array(image_resized, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            with st.spinner("Analyzing image..."):
                predictions = model.predict(image_array, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

            st.subheader("Prediction")
            st.write(f"**Predicted category:** {predicted_class}")
            st.write(f"**Model confidence:** {confidence:.2f}%")

            st.subheader("Class probabilities")
            for class_name, probability in zip(CLASS_NAMES, predictions):
                st.write(f"**{class_name}:** {probability * 100:.2f}%")
                st.progress(float(probability))

    except Exception as error:
        st.error("The image could not be processed.")
        st.exception(error)

st.divider()
st.caption(
    "Educational project only. This model is not clinically validated "
    "and must not be used to diagnose, rule out, or make treatment "
    "decisions about any medical condition. Consult a qualified "
    "medical professional for MRI interpretation."
)
