from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from datetime import datetime
from huggingface_hub import hf_hub_download
from keras.models import load_model
import keras


###############################################################################
# 1. APP SETUP
###############################################################################
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

###############################################################################
# 2. USER MODEL
###############################################################################
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

###############################################################################
# 3. AUTHENTICATION ROUTES
###############################################################################
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        
        flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == "POST":
        first_name = request.form.get('firstName')
        last_name = request.form.get('lastName')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        
        user = User(
            first_name=first_name,
            last_name=last_name,
            email=email
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Here you would typically:
            # 1. Generate a password reset token
            # 2. Send an email with the reset link
            # 3. Store the token in the database with an expiration time
            flash('Password reset instructions have been sent to your email.', 'success')
        else:
            flash('Email not found.', 'error')
    
    return render_template('forgot_password.html')

###############################################################################
# 4. MODEL PATHS AND VARIABLES
###############################################################################
# Paths to your Keras model files


###############################################################################
# 5. GLOBAL MODEL VARIABLES
###############################################################################
covid_model = None
pneumonia_model = None
tb_model = None
lung_model = None
alz_model = None

###############################################################################
# 6. MODEL LOADING FUNCTIONS
###############################################################################
###############################################################################
# 6. MODEL LOADING FUNCTIONS
###############################################################################
def load_covid_model_if_needed():
    global covid_model
    if covid_model is None:
        # Define model info
        HF_REPO_ID = "shrrynsh/COVID_Detect.keras"
        MODEL_FILENAME = "COVID_Detect.keras"
        MODEL_DIR = "models"
        
        # Ensure the models directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
        
        # Download the model if it's not already present
        if not os.path.exists(MODEL_PATH):
            print("Downloading COVID model from Hugging Face...")
            
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                local_dir=MODEL_DIR
            )
        
        # Load the model using keras.saving
        os.environ["KERAS_BACKEND"] = "tensorflow"
        
        covid_model = keras.saving.load_model(MODEL_PATH)
        print("COVID model loaded successfully")

def load_pneumonia_model_if_needed():
    global pneumonia_model
    if pneumonia_model is None:
        # Define model info
        HF_REPO_ID = "shrrynsh/pneumonia_predict.keras"
        MODEL_FILENAME = "pneumonia_predict.keras"
        MODEL_DIR = "models"
        
        # Ensure the models directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
        
        # Download the model if it's not already present
        if not os.path.exists(MODEL_PATH):
            print("Downloading pneumonia model from Hugging Face...")
            
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                local_dir=MODEL_DIR
            )
        
        # Load the model using keras.saving
        os.environ["KERAS_BACKEND"] = "tensorflow"
        
        pneumonia_model = keras.saving.load_model(MODEL_PATH)
        print("Pneumonia model loaded successfully")

def load_tb_model_if_needed():
    global tb_model
    if tb_model is None:
        # Define model info
        HF_REPO_ID = "shrrynsh/tb_model_final.keras"
        MODEL_FILENAME = "tb_model_final.keras"
        MODEL_DIR = "models"
        
        # Ensure the models directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
        
        # Download the model if it's not already present
        if not os.path.exists(MODEL_PATH):
            print("Downloading TB model from Hugging Face...")
            
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                local_dir=MODEL_DIR
            )
        
        # Load the model using keras.saving
        os.environ["KERAS_BACKEND"] = "tensorflow"
        
        tb_model = keras.saving.load_model(MODEL_PATH)
        print("TB model loaded successfully")

def load_lung_model_if_needed():
    global lung_model
    if lung_model is None:
        # Define model info
        HF_REPO_ID = "shrrynsh/best_model_lung_cancer.keras"
        MODEL_FILENAME = "best_model_lung_cancer.keras"
        MODEL_DIR = "models"
        
        # Ensure the models directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
        
        # Download the model if it's not already present
        if not os.path.exists(MODEL_PATH):
            print("Downloading lung cancer model from Hugging Face...")
            
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                local_dir=MODEL_DIR
            )
        
        # Load the model using keras.saving
        os.environ["KERAS_BACKEND"] = "tensorflow"
        
        lung_model = keras.saving.load_model(MODEL_PATH)
        print("Lung cancer model loaded successfully")

def load_alz_model_if_needed():
    global alz_model
    if alz_model is None:
        # Define model info
        HF_REPO_ID = "shrrynsh/alzheimer_detect.keras"
        MODEL_FILENAME = "alzheimer_detect.keras"
        MODEL_DIR = "models"
        
        # Ensure the models directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
        
        # Download the model if it's not already present
        if not os.path.exists(MODEL_PATH):
            print("Downloading Alzheimer's model from Hugging Face...")
            
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                local_dir=MODEL_DIR
            )
        
        # Load the model using keras.saving
        os.environ["KERAS_BACKEND"] = "tensorflow"
  
        alz_model = keras.saving.load_model(MODEL_PATH)
        print("Alzheimer's model loaded successfully")
###############################################################################
# 7. PREPROCESS & PREDICT LOGIC FOR THE FIRST 4 MODELS (COVID, PNEUMONIA, TB, LUNG)
###############################################################################
# -- COVID --
def preprocess_covid_image(file, img_size=(224,224)):
    pil_img = Image.open(file).convert('RGB')
    pil_img = pil_img.resize(img_size)
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)
    return img_array
    
def predict_covid(img_array):
    pred_prob = covid_model.predict(img_array)[0][0]
    if pred_prob < 0.5:
        return "COVID-19", float(1 - pred_prob)
    else:
        return "Non-COVID-19", float(pred_prob)

# -- PNEUMONIA --

def preprocess_pneumonia_image(file, img_size=(224,224)):
    pil_img = Image.open(file).convert('RGB')
    pil_img = pil_img.resize(img_size)
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_pneumonia(img_array):
    pred_prob = pneumonia_model.predict(img_array)[0][0]
    if pred_prob < 0.5:
        return "Pneumonia", float(1 - pred_prob)
    else:
        return "Normal", float(pred_prob)

# -- TB --
def preprocess_tb_image(file, img_size=(256,256)):
    pil_img = Image.open(file)
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
    pil_img = pil_img.resize(img_size)
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (256,256,1)
    img_array = np.expand_dims(img_array, axis=0)   # (1,256,256,1)
    return img_array


def predict_tb(img_array):
    pred_prob = tb_model.predict(img_array)[0][0]
    if pred_prob < 0.5:
        return "Normal", float(1 - pred_prob)
    else:
        return "TB Detected", float(pred_prob)

# -- LUNG (3-class) --
lung_class_names = ["Normal", "Malignant", "Benign"]

def preprocess_lung_image(file, img_size=(224,224)):
    pil_img = Image.open(file).convert('RGB')
    pil_img = pil_img.resize(img_size)
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_lung_cancer(file):
    load_lung_model_if_needed()
    processed_image = preprocess_lung_image(file)
    prediction = lung_model.predict(processed_image, verbose=0)
    predicted_index = np.argmax(prediction, axis=1)[0]
    confidence = float(prediction[0][predicted_index])
    predicted_label = lung_class_names[predicted_index]
    return predicted_label, confidence, prediction.tolist()

###############################################################################
# 8. ALZHEIMER DETECTION LOGIC
###############################################################################
alz_label_mapping = {
    0: "Mild_Demented",
    1: "Moderate_Demented",
    2: "Non_Demented",
    3: "Very_Mild_Demented"
}

def preprocess_alz_image(image_data, target_size=(128,128)):
    """
    1) Converts raw bytes to a NumPy array
    2) Decodes with OpenCV in grayscale
    3) Resizes to (224,224)
    4) Normalizes to [0..1]
    5) Expands dims => (1,224,224,1)
    """
    # Convert bytes => 1D np array
    arr = np.frombuffer(image_data, dtype=np.uint8)
    # Decode in grayscale
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("cv2.imdecode returned None; invalid or corrupted image data.")

    # Resize
    img = cv2.resize(img, target_size)
    # Normalize
    img = img / 255.0
    # Expand dims => (1,224,224,1)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict_alzheimers(img_array):
    """
    Model outputs a 4D probability vector: [class0, class1, class2, class3].
    We pick the highest-prob class, map it to a string, and return confidence.
    """
    preds = alz_model.predict(img_array)[0]  # shape: (4,)
    class_idx = np.argmax(preds)
    confidence = float(preds[class_idx])
    predicted_label = alz_label_mapping[class_idx]
    return predicted_label, confidence, preds

###############################################################################
# 9. PREDICTION ENDPOINTS
###############################################################################
@app.route("/")
@login_required
def home():
    return render_template("index.html")

@app.route("/profile")
@login_required
def profile():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    
    # Calculate statistics
    total_predictions = len(predictions)
    covid_predictions = len([p for p in predictions if p.model_type == 'COVID-19'])
    pneumonia_predictions = len([p for p in predictions if p.model_type == 'Pneumonia'])
    tb_predictions = len([p for p in predictions if p.model_type == 'TB'])
    
    return render_template('profile.html',
                         user=current_user,
                         predictions=predictions,
                         total_predictions=total_predictions,
                         covid_predictions=covid_predictions,
                         pneumonia_predictions=pneumonia_predictions,
                         tb_predictions=tb_predictions)

@app.route("/predict_covid", methods=["POST"])
@login_required
def predict_covid_route():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Preprocess the image
        img_array = preprocess_covid_image(file)
        
        # Make prediction
        label, confidence = predict_covid(img_array)
        
        # Store prediction in database
        prediction = Prediction(
            user_id=current_user.id,
            model_type='COVID-19',
            result=label,
            confidence=confidence
        )
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify({
            "prediction": label,
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error in COVID prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_pneumonia", methods=["POST"])
@login_required
def predict_pneumonia_route():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Preprocess the image
        img_array = preprocess_pneumonia_image(file)
        
        # Make prediction
        label, confidence = predict_pneumonia(img_array)
        
        # Store prediction in database
        prediction = Prediction(
            user_id=current_user.id,
            model_type='Pneumonia',
            result=label,
            confidence=confidence
        )
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify({
            "prediction": label,
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error in Pneumonia prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_tb", methods=["POST"])
@login_required
def predict_tb_route():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Preprocess the image
        img_array = preprocess_tb_image(file)
        
        # Make prediction
        label, confidence = predict_tb(img_array)
        
        # Store prediction in database
        prediction = Prediction(
            user_id=current_user.id,
            model_type='TB',
            result=label,
            confidence=confidence
        )
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify({
            "prediction": label,
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error in TB prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_lung", methods=["POST"])
@login_required
def predict_lung_route():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Make prediction
        label, confidence, probabilities = predict_lung_cancer(file)
        
        # Store prediction in database
        prediction = Prediction(
            user_id=current_user.id,
            model_type='Lung Cancer',
            result=label,
            confidence=confidence
        )
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify({
            "prediction": label,
            "confidence": confidence,
            "probabilities": probabilities
        })
    except Exception as e:
        print(f"Error in Lung Cancer prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_alz", methods=["POST"])
@login_required
def predict_alz_route():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Read and preprocess the image
        image_data = file.read()
        img_array = preprocess_alz_image(image_data)
        
        # Make prediction
        label, confidence, probabilities = predict_alzheimers(img_array)
        
        # Store prediction in database
        prediction = Prediction(
            user_id=current_user.id,
            model_type='Alzheimer\'s',
            result=label,
            confidence=confidence
        )
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify({
            "prediction": label,
            "confidence": confidence,
            "probabilities": probabilities
        })
    except Exception as e:
        print(f"Error in Alzheimer's prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/project_success")
@login_required
def project_success():
    # Get all predictions
    predictions = Prediction.query.all()
    
    # Calculate total predictions
    total_predictions = len(predictions)
    
    # Calculate predictions by model
    covid_predictions = len([p for p in predictions if p.model_type == 'COVID-19'])
    pneumonia_predictions = len([p for p in predictions if p.model_type == 'Pneumonia'])
    tb_predictions = len([p for p in predictions if p.model_type == 'TB'])
    
    # Calculate success rates (based on confidence > 0.8)
    covid_success = len([p for p in predictions if p.model_type == 'COVID-19' and p.confidence > 0.8])
    pneumonia_success = len([p for p in predictions if p.model_type == 'Pneumonia' and p.confidence > 0.8])
    tb_success = len([p for p in predictions if p.model_type == 'TB' and p.confidence > 0.8])
    
    # Calculate success rates as percentages
    covid_success_rate = round((covid_success / covid_predictions * 100) if covid_predictions > 0 else 0, 1)
    pneumonia_success_rate = round((pneumonia_success / pneumonia_predictions * 100) if pneumonia_predictions > 0 else 0, 1)
    tb_success_rate = round((tb_success / tb_predictions * 100) if tb_predictions > 0 else 0, 1)
    
    return render_template('project_success.html',
                         total_predictions=total_predictions,
                         covid_predictions=covid_predictions,
                         pneumonia_predictions=pneumonia_predictions,
                         tb_predictions=tb_predictions,
                         covid_success_rate=covid_success_rate,
                         pneumonia_success_rate=pneumonia_success_rate,
                         tb_success_rate=tb_success_rate)

###############################################################################
# 8. INITIALIZE DATABASE
###############################################################################
def init_db():
    with app.app_context():
        db.create_all()
        # Load all models
        try:
            load_covid_model_if_needed()
            load_pneumonia_model_if_needed()
            load_tb_model_if_needed()
            load_lung_model_if_needed()
            load_alz_model_if_needed()
            print("All models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5500)