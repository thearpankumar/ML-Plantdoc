import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import requests
import json
import serial
import time



# Function to read the last line from the file
def read_last_line():
    try:
        with open('../moisture_levels.txt', 'r') as file:
            lines = file.readlines()
            if lines:
                return lines[-1].strip()  # Return the last line without newline characters
    except FileNotFoundError:
        return None
    return None

# Function to make the API request
def get_jazz_subcategories(api_key, desease, moisture):
    class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={api_key}"
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"give precautation for preventing {class_name[result_index]}, also my moisture level is {moisture}% if it good for the plant , also give response in {selected_option}?"}
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def custom_querry(api_key, querry):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={api_key}"
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": " {} ?".format(querry)}
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition","rice Crop Disease prediction"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    options = ["ENGLISH", "HINDI", "TAMIL"]

    # Create a dropdown box
    selected_option = st.selectbox("Select your Language :", options, index=0)

    # Replace with your actual API key
    API_KEY = "AIzaSyAgLVNdRob3KZLeR-ZYd_8MHYc1G9BdOYo"

    # Define the endpoint URL
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={API_KEY}"
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image, width=4, use_column_width=True)
    
    # Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        moisture_level = read_last_line()
        if moisture_level:
            print(f"Soil Moisture Level: {moisture_level}%")
        with st.spinner("Fetching data..."):
            try:
                response = get_jazz_subcategories(API_KEY, result_index, moisture_level)
                
                if 'candidates' in response and response['candidates']:
                    content = response['candidates'][0]['content']['parts'][0]['text']
                    st.subheader("Response:")
                    st.write(content)
                else:
                    st.error("No valid response received.")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
        
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                     'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                     'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                     'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                     'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                     'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                     'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                     'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                     'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                     'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                     'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                     'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                     'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                     'Tomato___healthy']
        try:
            st.success(f"Model is Predicting it's a {class_name[result_index]}, Your SOIL MOSTURE (%) {moisture_level} ?")
        except Exception as e:
            pass

    # Custom Query Input
    st.subheader("Ask a Custom Query")
    custom_query = st.text_input("Enter your custom query here:")
    
    if st.button("Submit Query"):
        with st.spinner("Processing your query..."):
            try:
                custom_response = custom_querry(API_KEY, custom_query)
                
                if 'candidates' in custom_response and custom_response['candidates']:
                    content = custom_response['candidates'][0]['content']['parts'][0]['text']
                    st.subheader("Response to your query:")
                    st.write(content)
                else:
                    st.error("No valid response received.")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
