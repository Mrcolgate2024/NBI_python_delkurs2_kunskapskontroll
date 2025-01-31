import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import cv2
import warnings
import time
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import altair as alt

# Canvas and Image Settings
IMG_SIZE = 28  # Target size for the model (28x28)

# =============================================================================
# Creating the streamlit application. #
# =============================================================================


# ========================== Sidebar Navigation ==========================
# Sidebar for navigation
st.sidebar.title("🧠 AI-Powered Digit Recognition")

# Centered profile picture and name
st.sidebar.image('IMG_6025.jpg', use_container_width=True, caption="")
st.sidebar.markdown("""
**👨‍💻 Björn R Axelsson**  
AI & ML Engineer""")

st.sidebar.header("Input Selection")
nav = st.sidebar.radio("🔍 Choose Image Input",["Upload image", "Draw image", "Camera image"])
model_choice = st.sidebar.selectbox("🛠️ Choose Model Input", ["Extra Trees"])

# About the Model section
st.sidebar.header("📊 About the Model")
st.sidebar.write("""
    The AI-Powered Digit Recognition solution is a machine learning model designed to predict handwritten digits from 0 to 9. It has been trained on the MNIST dataset, which contains 70,000 handwritten digit images, ensuring accurate and reliable predictions.
    
**Model Details**
- 🚀 Solutions: Scikit-Learn, Python, Streamlit
- 🏆 Model: Extra Trees Classifier
- 🎯 Accuracy Score: **97.15%**
""")

# Contact information
st.sidebar.header("📬 Contact Details")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/bjorn-r-axelsson/)")
st.sidebar.write("[GitHub](https://github.com/Mrcolgate2024)")
st.sidebar.write("[Email](bjorn.r.axelsson@gmail.com)")


# ========================== Load Models ==========================
# Load Data and Model
def load_models():
    # Get the current directory (same directory as the script)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative paths to the model files
    extratree_model_path = os.path.join(current_dir, 'extra_trees_clf.pkl')
    voting_model_path = os.path.join(current_dir, 'mnist_voting_clf.pkl')
    scaler_model_path = os.path.join(current_dir, 'mnist_scaler.pkl')

    # Load the Extra Trees and Voting Classifier models
    extratree_model = joblib.load(extratree_model_path)
    voting_model = joblib.load(voting_model_path)
    scaler_model = joblib.load(scaler_model_path)

    return extratree_model, voting_model, scaler_model

# Load the models using the defined function
extratree_model, voting_model, scaler_model = load_models()


# ========================== Image Preprocessing ==========================
# Image Preprocessing Function
def preprocess_image(image):
    # Step 1: Convert to grayscale (if not already)
    if len(image.shape) == 3:  # If the image has color channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # # Step 1: Histogram-based thresholding (Otsu's method)
    # Otsu's method automatically finds the optimal threshold value
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    
    # Step 2: Resize the image to 28x28 pixels
    image_resized = cv2.resize(thresholded_image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Step 3: Invert colors (black digits on white background)
    inverted_image = 255 - image_resized
    
    # Step 4: Flatten the image to a 1D array and normalize
    flat_image = inverted_image.flatten().astype('float32')
    
    return flat_image.reshape(1, -1)  # Return as 1D array for the model

# Helper function to convert numpy array to DataFrame
def np_to_df(np_array):
    return pd.DataFrame(np_array)


# ========================== UPLOAD IMAGE OPTION ==========================
if nav == "Upload image":
    # Main section for the page
    st.title("📁 Predict using uploaded image")
    st.write("Upload a digit image to predict the number")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:
        # Convert uploaded file to an OpenCV-readable format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read in color (OpenCV default)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Preprocess the image for the model
        processed_image = preprocess_image(image)

        # Predict the digit using the selected model
        if model_choice == "Extra Trees":
            prediction = extratree_model.predict(processed_image)
        else:
            prediction = voting_model.predict(processed_image)
        
        predicted_digit = prediction[0]

        # Display the predicted digit
        st.write(f"**Predicted Digit:** {predicted_digit}")

        # Handle probability outputs
        if model_choice == "Extra Trees":
            outputs = extratree_model.predict_proba(processed_image).squeeze()
            percent_outputs = outputs * 100

            # Create a DataFrame with probabilities
            chart_data = pd.DataFrame(percent_outputs, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], columns=['Probability'])
            
            # Create the bar chart with Altair
            bars = alt.Chart(chart_data.reset_index()).mark_bar().encode(
                x=alt.X('index:N', title='Digits'),  # X axis with label "Digits"
                y='Probability:Q',  # Y axis for the probability values (percentages)
                color=alt.Color('index:N', scale=alt.Scale(scheme='blues'), legend=None),  # No legend
                tooltip=['index:N', 'Probability:Q']
            )
            
            # Add text labels on top of the bars
            labels = bars.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,  # Distance between the bar and the label
                color='black'
            ).encode(
                text=alt.Text('Probability:Q', format='.0f')  # Remove decimals (show as whole number)
            )
            
            # Combine the bars and the labels into one chart
            chart = bars + labels
            
            # Display the chart
            st.altair_chart(chart, use_container_width=True)
        
        elif model_choice == "Voting Classifier":
            # No probability chart for the voting classifier
            st.write("Note: The Voting Classifier with SVM does not provide probabilities, so no chart is available.")
            
if nav == "Draw image":
    # Main section for the page
    st.title("✏️ Predict using drawn image")
    st.write("Draw an image from 0 to 9 to predict the number")
    
    stroke_width = st.slider("Stroke width: ", 1, 50, 20)
    
    # Canvas for drawing (Black Background, White Text)
    canvas = st_canvas(
        stroke_width=stroke_width,
        stroke_color="#fff",
        background_color="#000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )


    # Preprocess the drawn image
    if canvas.image_data is not None:
        image = Image.fromarray((canvas.image_data[:, :, 0]).astype(np.uint8))
        image = image.resize((28, 28))

        # Prepare image for prediction (scikit-learn model)
        array = np.array(image)
        array = array.reshape(1, 784)
        
        if model_choice == "Extra Trees":
            outputs = extratree_model.predict_proba(array).squeeze()  # Get probability predictions
        else:
            outputs = voting_model.predict(array)  # Predict using the voting classifier
        
        # Display progress bar while predicting
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        
       # Find the predicted label with the highest probability
        ind_max = np.argmax(outputs)
        
        # Display prediction result with correct label
        st.markdown(f"<h3 style='text-align: center;'>Predicted Digit: {ind_max}</h3>", unsafe_allow_html=True)
        
        if model_choice == "Extra Trees":
            # Convert the outputs to percentages
            percent_outputs = outputs * 100
            chart_data = pd.DataFrame(percent_outputs, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], columns=['Probability'])
            
            # Create the bar chart with Altair
            bars = alt.Chart(chart_data.reset_index()).mark_bar().encode(
                x=alt.X('index:N', title='Digits'),
                y='Probability:Q',
                color=alt.Color('index:N', scale=alt.Scale(scheme='blues'), legend=None),
                tooltip=['index:N', 'Probability:Q']
            )
            
            # Add text labels on top of the bars
            labels = bars.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black'
            ).encode(
                text=alt.Text('Probability:Q', format='.0f')
            )
            
            # Combine the bars and the labels into one chart
            chart = bars + labels
            
            # Display the chart
            st.altair_chart(chart, use_container_width=True)
        
        elif model_choice == "Voting Classifier":
            # No probability chart for the voting classifier
            st.write("Note: The Voting Classifier with SVM does not provide probabilities, so no chart is available.")
            

if nav == "Camera image":
    # Main section for the page
    st.title("📸 Predict using camera image")
    st.write("Take a picture of a digit using the camera on the computer to predict the number")

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the camera is opened correctly
    if not cap.isOpened():
        st.write("Error: Could not open webcam.")
    else:
        # Capture an image when the button is clicked
        if st.button("Capture Image"):
            # Capture a frame
            ret, frame = cap.read()
            if ret:
                # Display the captured image
                st.image(frame, caption="Captured Image", use_container_width=True)

                # Preprocess the captured image
                processed_image = preprocess_image(frame)

                # Predict the digit using the selected model
                if model_choice == "Extra Trees":
                    prediction = extratree_model.predict(processed_image)
                else:
                    prediction = voting_model.predict(processed_image)
                
                predicted_digit = prediction[0]

                # Display the predicted digit
                st.write(f"**Predicted Digit:** {predicted_digit}")

                # Handle probability outputs
                if model_choice == "Extra Trees":
                    outputs = extratree_model.predict_proba(processed_image).squeeze()
                    percent_outputs = outputs * 100

                    # Create a DataFrame with probabilities
                    chart_data = pd.DataFrame(percent_outputs, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], columns=['Probability'])
                    
                    # Create the bar chart with Altair
                    bars = alt.Chart(chart_data.reset_index()).mark_bar().encode(
                        x=alt.X('index:N', title='Digits'),
                        y='Probability:Q',
                        color=alt.Color('index:N', scale=alt.Scale(scheme='blues'), legend=None),
                        tooltip=['index:N', 'Probability:Q']
                    )
                    
                    # Add text labels on top of the bars
                    labels = bars.mark_text(
                        align='center',
                        baseline='bottom',
                        dy=-5,
                        color='black'
                    ).encode(
                        text=alt.Text('Probability:Q', format='.0f')
                    )
                    
                    # Combine the bars and the labels into one chart
                    chart = bars + labels
                    
                    # Display the chart
                    st.altair_chart(chart, use_container_width=True)
                
                elif model_choice == "Voting Classifier":
                    # No probability chart for the voting classifier
                    st.write("Note: The Voting Classifier with SVM does not provide probabilities, so no chart is available.")

        # Release the camera when done
        cap.release()

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
