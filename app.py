# import tensorflow as tf
# import numpy as np
# import matplotlib as mpl
# import streamlit as sl
# import os
# import seaborn as sns
# print(sns.__version__)
# print(os.path)
# print(sl.__version__)
# print(mpl.__version__)
# print(np.__version__)
# print(tf.__version__)

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model('cnn_model.h5')


# Define the class labels (replace with your actual class labels)
class_labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'} 

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to the input size expected by the model
    img = np.array(img)
    #img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make predictions
def predict(image):
    img = preprocess_image(image)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    return class_labels[predicted_class[0]]

# Streamlit app
st.image("banner.png", use_container_width=True)
st.title("Fruits and Vegetables Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        label = predict(image)
        st.markdown(f"<h2 style='color: green; text-align: center;'>Prediction: {label}</h2>", unsafe_allow_html=True)
        st.image(image, caption='Uploaded Image.', use_container_width=False, width=700)
        st.write("")
    except ValueError:
        st.markdown(f"<h2 style='color: blue; text-align: center;'>It is neither a Fruit nor a Vegetable</h2>", unsafe_allow_html=True)

# if uploaded_file is not None:
#     try:
#         image = Image.open(uploaded_file)
#         label = predict(image)
#         st.markdown(f"<h2 style='color: green; text-align: center;'>Prediction: {label}</h2>", unsafe_allow_html=True)
#         st.markdown(
#             f"""
#             <div style='display: flex; justify-content: center;'>
#                 <img src='data:image/jpg;base64,{image}' width='500'>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#         st.write("")
#     except ValueError:
#         st.markdown("<h2 style='color: green;'>It is neither a Fruit nor a Vegetable</h2>", unsafe_allow_html=True)

# else:
#     st.markdown("<h2 style='color: green;'>This is not a fruit nor vegetable</h2>", unsafe_allow_html=True)
