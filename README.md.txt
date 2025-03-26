# 🚗 Drowsiness Detection System  

A real-time Drowsiness Detection System using OpenCV, CNN, SVM, and Sound Alerts. This project detects if a person is drowsy or alert using a webcam feed and triggers an alarm when drowsiness is detected.  

📌 Features  
✅ Real-time Drowsiness Detection using CNN & SVM  
✅ Face & Eye Tracking using OpenCV  
✅ Sound Alert System (Plays `alert.wav` when drowsy)  
✅ Customizable Threshold for Alert vs. Drowsy  
✅ User-friendly Interface


 🛠️ Installation  
🔹 Step 1: Clone the Repository
git clone https://github.com/YOUR_USERNAME/drowsiness-detection.git
cd drowsiness-detection

🔹 Step 2: Install Dependencies
Run the following command to install all required packages:
pip install -r requirements.txt

🔹 Step 3: Ensure You Have the Alert Sound File
Place an alert sound (alert.wav) inside the sounds/ folder.
If you don't have one, you can download a beep sound from this link or create one using any audio editor.

🚀 How to Run the System
🔹 Start Real-Time Drowsiness Detection

Run the following command:
python app.py
Press Q to exit the program.
The system will detect drowsiness and play a sound alert if needed.

🏋️‍♂️ How to Train the Model:
🔹 Train the CNN Model
If you need to retrain the CNN model, run:

python scripts/train_cnn.py

This will save the trained model as:
models/cnn_drowsiness.h5

🔹 Train the SVM Model
To train the SVM model, run:

python scripts/train_svm.py

This will save the trained model as:
models/svm_drowsiness.joblib
