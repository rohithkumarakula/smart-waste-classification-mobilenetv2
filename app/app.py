from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# Keep class names in the same order used to train the model.
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_SIZE = (224, 224)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_waste_classifier.keras"


# Set up the browser tab and page layout.
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add a clean color palette and a few reusable card styles.
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #f3f8f5 0%, #f8fafc 55%, #eef5f1 100%);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stMain"] .stMarkdown h1,
    [data-testid="stMain"] .stMarkdown h2,
    [data-testid="stMain"] .stMarkdown h3,
    [data-testid="stMain"] .stMarkdown h4,
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4 {
        color: #173a2b !important;
    }
    [data-testid="stMain"] .stMarkdown p,
    [data-testid="stMain"] p,
    [data-testid="stMain"] [data-testid="stWidgetLabel"],
    [data-testid="stMain"] [data-testid="stWidgetLabel"] *,
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stCaptionContainer"] *,
    [data-testid="stMain"] [data-testid="stAlert"] *,
    [data-testid="stMain"] [data-testid="stStatusWidget"] * {
        color: #374151 !important;
    }
    [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {
        color: #374151 !important;
        background-color: #ffffff;
        border-color: #dcebe1;
    }
    [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] h1,
    [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] h2,
    [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] h3,
    [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] h4 {
        color: #173a2b !important;
    }
    [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] p {
        color: #374151 !important;
    }
    [data-testid="stSidebar"] {
        background: #10251d;
    }
    [data-testid="stSidebar"] * {
        color: #eef7f1;
    }
    .hero-card {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        color: #374151;
        background: linear-gradient(120deg, #e5f3e9 0%, #d4eadb 58%, #c6e2d0 100%);
        box-shadow: 0 16px 38px rgba(24, 112, 75, 0.18);
        margin-bottom: 1.5rem;
    }
    .hero-card h1 {
        color: #173a2b !important;
        font-size: clamp(2rem, 5vw, 3.2rem);
        margin: 0 0 0.6rem 0;
        letter-spacing: -0.04em;
    }
    .hero-card p {
        color: #374151 !important;
        font-size: 1.08rem;
        margin: 0;
        max-width: 760px;
    }
    .section-heading {
        color: #173a2b !important;
        margin-top: 1rem;
    }
    .result-card {
        padding: 1.4rem 1.6rem;
        border: 1px solid #dcebe1;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.88);
        color: #374151 !important;
        box-shadow: 0 8px 24px rgba(21, 54, 38, 0.06);
    }
    .result-label {
        color: #374151 !important;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .result-category {
        color: #173a2b !important;
        font-size: clamp(1.8rem, 4vw, 2.6rem);
        font-weight: 750;
        margin: 0.25rem 0 0.5rem 0;
    }
    div[data-testid="stFileUploader"] {
        border: 1px dashed #85b99b;
        border-radius: 16px;
        padding: 0.5rem;
        background: rgba(255, 255, 255, 0.7);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load the saved model once and reuse it between Streamlit interactions.
@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, compile=False)


# Explain the project and model in the sidebar.
with st.sidebar:
    st.markdown("## ♻️ Project")
    st.write("**Smart Waste Classification Using MobileNetV2**")
    st.caption(
        "An image-recognition demo that predicts which waste category an item "
        "most resembles."
    )
    st.divider()
    st.markdown("### Model architecture")
    st.write("MobileNetV2 pretrained on ImageNet")
    st.write("Feature extractor: `include_top=False`")
    st.write("Global average pooling → dropout (0.2) → 6-class softmax")
    st.write("Input: `224 × 224 × 3` RGB image")
    st.divider()
    st.markdown("### Supported image formats")
    st.write("JPG · JPEG · PNG")


# Introduce the dashboard.
st.markdown(
    """
    <div class="hero-card">
        <h1>Smart Waste Classification</h1>
        <p>AI-powered image recognition to help identify everyday waste and
        support more informed sorting decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stop with a clear message if the trained model has not been created.
if not MODEL_PATH.is_file():
    st.error(
        f"Trained model not found at `{MODEL_PATH}`. "
        "Train the model first, then restart this app."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except (OSError, ValueError) as error:
    st.error(f"The trained model could not be loaded: {error}")
    st.stop()

# Let the user upload an image and show a preview.
st.markdown("## Classify an image")
uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"],
    help="Upload one JPG, JPEG, or PNG image to classify.",
)

if uploaded_file is None:
    st.info("Upload an image to see the model's prediction and class probabilities.")
else:
    try:
        # Decode the upload, convert it to RGB, and prepare model-sized pixels.
        original_image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
        resized_image = original_image.resize(IMAGE_SIZE)
        image_array = np.asarray(resized_image, dtype=np.float32)
        image_batch = np.expand_dims(image_array, axis=0)

        # Apply the same MobileNetV2 preprocessing used during training.
        processed_batch = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_batch
        )

        # Predict probabilities for the six categories.
        predictions = model.predict(processed_batch, verbose=0)
        if predictions.shape != (1, len(CLASS_NAMES)):
            st.error(
                "The model returned an unexpected output shape. "
                f"Expected (1, {len(CLASS_NAMES)}), got {predictions.shape}."
            )
            st.stop()

        probabilities = predictions[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_category = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index])

        # Present the uploaded image beside the model's result.
        image_column, result_column = st.columns([1, 1], gap="large")
        with image_column:
            st.image(original_image, caption="Uploaded image", use_container_width=True)
        with result_column:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted category</div>
                    <div class="result-category">{predicted_category.title()}</div>
                    <div>Model confidence: <strong>{confidence:.1%}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(confidence, text=f"Confidence: {confidence:.1%}")

        # Display the model's actual probability for every class.
        st.markdown("### Prediction probabilities")
        probability_chart = {
            class_name.title(): float(probabilities[index])
            for index, class_name in enumerate(CLASS_NAMES)
        }
        st.bar_chart(probability_chart, y_label="Probability")

    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"That image could not be opened. Please upload a valid JPG, JPEG, or PNG: {error}")

# Describe each category to make the predictions easier to understand.
st.markdown('<h2 class="section-heading">Waste categories</h2>', unsafe_allow_html=True)
category_descriptions = {
    "Cardboard": "Boxes and other clean, corrugated or paperboard packaging.",
    "Glass": "Bottles and jars made primarily of glass.",
    "Metal": "Metal cans and other recyclable metal packaging.",
    "Paper": "Newspapers, sheets, and other paper products.",
    "Plastic": "Plastic bottles, containers, and packaging.",
    "Trash": "Mixed or non-recyclable waste that does not fit the other categories.",
}
description_columns = st.columns(3)
for index, (category, description) in enumerate(category_descriptions.items()):
    with description_columns[index % 3]:
        with st.container(border=True):
            st.markdown(f"### {category}")
            st.write(description)

st.caption(
    "Predictions are model estimates. Check local recycling guidance before sorting."
)