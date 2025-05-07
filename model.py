from PIL import Image
import numpy as np
import joblib
from sklearn.exceptions import NotFittedError

# Preprocessing classes
class GrayscaleConverter:
    def __init__(self, max_images=1):
        self.max_images = max_images
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [Image.open(img_path).convert('L') for img_path in X]

class DFTProcessor:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        def normalize_dft(img):
            img_array = np.array(img)
            padded = np.pad(img_array, ((0, img_array.shape[0]), (0, img_array.shape[1])), mode='constant')
            fourier = np.fft.fft2(padded)
            return np.abs(fourier)**2 / (padded.size**2)
        return [normalize_dft(img) for img in X]

class AzimuthalAverager:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        def azimuthal_average(spectrum):
            h, w = spectrum.shape
            center = (h // 2, w // 2)
            y, x = np.indices((h, w))
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            r = r.astype(int)
            radial_mean = np.bincount(r.ravel(), spectrum.ravel()) / np.bincount(r.ravel())
            return radial_mean[:min(h, w) // 2]
        return [azimuthal_average(spectrum) for spectrum in X]

# SVMClassifier class
class SVMClassifier:
    def __init__(self):
        self.classifier = None  # Load your trained SVC here
    def decision_function(self, X):
        try:
            return self.classifier.decision_function(X)
        except NotFittedError as e:
            raise Exception(f"Model not fitted: {str(e)}")

# Load the trained model (adjust path to your saved model)
svm_classifier = SVMClassifier()
try:
    svm_classifier.classifier = joblib.load('d_svm_model(1).pkl')  # Update with your actual path
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file 'svm_model.pkl' not found. Please check the path.")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Prediction function
def predict_image(image_path):
    try:
        # Preprocess the image
        grayscale = GrayscaleConverter(max_images=1).transform([image_path])[0]
        dft = DFTProcessor().transform([grayscale])[0]
        X_test = AzimuthalAverager().transform([dft])[0].reshape(1, -1)
        
        # Match feature length
        expected_length = 256  # Adjust based on your training data
        if X_test.shape[1] < expected_length:
            padding = np.zeros((1, expected_length - X_test.shape[1]))
            X_test = np.hstack((X_test, padding))
        elif X_test.shape[1] > expected_length:
            X_test = X_test[:, :expected_length]
        
        # Predict
        decision_scores = svm_classifier.decision_function(X_test)
        prediction = 1 if decision_scores[0] > 0.0 else 0  # Adjusted threshold
        result = f"Prediction: {'Fake' if prediction == 1 else 'Real'}<br>Confidence: {abs(decision_scores[0]):.4f}"
        return result
    except Exception as e:
        return f"Error processing image: {str(e)}"