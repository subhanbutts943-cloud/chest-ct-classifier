
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Chest CT Classifier",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 Chest CT Image Classifier")
st.write("Upload a chest CT image to see the model prediction.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("chest_ct_model.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Chest CT Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded CT Image",
        use_container_width=True
    )

    if st.button("Predict Image", type="primary"):
        img_resized = img.resize((224, 224))

        img_array = np.array(
            img_resized,
            dtype=np.float32
        ) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing image..."):
            predictions = model.predict(
                img_array,
                verbose=0
            )[0]

        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        st.subheader("Prediction Result")
        st.success(f"Predicted Class: {predicted_class}")
        st.write(f"Model confidence: {confidence:.2f}%")

        st.caption(
            "Educational demonstration only. "
            "This result is not a medical diagnosis."
        )