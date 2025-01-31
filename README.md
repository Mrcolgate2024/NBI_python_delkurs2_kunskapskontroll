# NBI_python_delkurs2_kunskapskontroll-
🧠 MNIST Digit Recognition App
A sleek and interactive AI-powered digit recognition app built using Python, Streamlit, OpenCV, and Scikit-Learn.

🚀 Features
✅ Upload, Draw, or Use Camera – Three ways to input a digit
✅ Camera Feed – Capture image using your camera on the computer
✅ Interactive Charts – Visualize model confidence levels
✅ Adaptive Thresholding – Improves digit recognition accuracy
✅ Smooth UI & Animations – Modern sidebar, CSS enhancements, and loading effects

📂 How It Works
1️⃣ User Chooses an Input Method:
Upload Image – Upload a handwritten digit image
Draw Image – Use an interactive canvas to draw a digit
Camera Image – Capture a digit using the webcam

2️⃣ Image Processing:
Converts the image to grayscale
Applies adaptive thresholding for better contrast
Resizes to 28×28 pixels (MNIST format)
Inverts colors (black digit on white background)
Normalizes and flattens the image for model input

3️⃣ Prediction:
The Extra Trees Classifier predicts the digit
Displays confidence levels in an interactive bar chart

4️⃣ Results:
The predicted digit is displayed with a success message
A probability distribution chart visualizes confidence scores

🛠️ Technologies Used
Python (Core logic)
Streamlit (Web app framework)
OpenCV (Image processing)
Scikit-Learn (Machine learning models)
Altair (Data visualization)
Pillow (Image manipulation)
Joblib (Model loading)

💻 How to Run
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/Mrcolgate2024/mnist-digit-recognition.git
cd mnist-digit-recognition

2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt

3️⃣ Run the App
bash
Copy
Edit
streamlit run app.py

📊 Example Output
Input Method	Prediction	Confidence (%)
Upload Image	7	98.2%
Draw Image	3	95.5%
Camera Image	9	92.7%

🎯 Future Improvements
🔹 Add more machine learning models for comparison
🔹 Enhance UI with animations and custom themes
🔹 Deploy the app online for public access

🚀 Built with ❤️ by Bjorn R Axelsson
