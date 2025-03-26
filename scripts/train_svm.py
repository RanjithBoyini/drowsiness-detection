import os
import numpy as np
from scipy.spatial import distance
from sklearn.svm import SVC
from joblib import dump

# ✅ Get the correct project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Moves up from 'scripts' folder
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure 'models/' exists

MODEL_PATH = os.path.join(MODEL_DIR, "svm_drowsiness.joblib")

# ✅ Dummy training data (Replace with real data)
X = [[0.3], [0.2], [0.5], [0.4]]  # Example EAR values
y = [0, 0, 1, 1]  # Labels: 0 = Alert, 1 = Drowsy

# ✅ Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X, y)

# ✅ Save the trained model
dump(svm_model, MODEL_PATH)

print(f"\n✅ SVM Model Trained and Saved at: {MODEL_PATH}")
