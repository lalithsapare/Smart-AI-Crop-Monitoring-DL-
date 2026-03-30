import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.set_page_config(page_title="🚜 Smart Agri AI", layout="wide")

@st.cache_resource
def load_all_models():
    model_paths = {
        'nutrient': "models/banana_nutrient_model.keras",
        'tomato': "models/tomato_disease_model.keras",
        'rice_disease': "models/rice_disease_model.keras",
        'rice_pest': "models/rice_pest_model.keras"
    }
    models = {}
    class_names = {
        'nutrient': ['Healthy', 'Nitrogen', 'P', 'K'],
        'tomato': ['Healthy', 'Blight', 'Septoria'],
        'rice_disease': ['Healthy', 'Blight', 'Spot'],
        'rice_pest': ['Healthy', 'Borer', 'Folder']
    }
    return models, class_names

models, class_names = load_all_models()

st.title("🚜 **Smart Agri AI - 7 Models**")
model_type = st.selectbox("Select Model:", ["Nutrient Deficiency", "Tomato Disease", "Rice Disease", "Rice Pest"])
uploaded_file = st.file_uploader("Upload leaf image", type=['jpg', 'png'])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, use_column_width=True)
    with col2:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        pred = models[model_type.lower().replace(' ', '_')].predict(img_array)[0]
        top_class = class_names[model_type.lower().replace(' ', '_')][np.argmax(pred)]
        confidence = np.max(pred) * 100
        
        st.success(f"**Result:** {top_class}")
        st.metric("Confidence", f"{confidence:.1f}%")
        st.progress(confidence/100)