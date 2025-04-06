# InVisionDX


![login](https://github.com/user-attachments/assets/39b9d3a7-7010-490f-9887-96a403e986fb)

A comprehensive Flask-based web application for automated medical diagnosis through image analysis. This platform leverages machine learning models to predict various medical conditions, all while offering a secure and modern user experience.

Access the ML models here--> [https://huggingface.co/shrrynsh
](https://huggingface.co/shrrynsh)
## Index

- [Project Overview](#project-overview)
- [Key Features](#key-features)
  - [User Authentication System](#user-authentication-system)
  - [Medical Image Analysis](#medical-image-analysis)
  - [Machine Learning Integration](#machine-learning-integration)
  - [Database Management](#database-management)
  - [Frontend Features](#frontend-features)
  - [Technical Stack](#technical-stack)
  - [Security Features](#security-features)
- [Notable Implementation Details](#notable-implementation-details)
  - [Image Processing Pipeline](#image-processing-pipeline)
  - [Prediction System](#prediction-system)
  - [User Experience](#user-experience)
  - [Data Management](#data-management)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)



## Project Overview

This web application is designed to analyze medical images and predict conditions such as COVID-19, Pneumonia, Tuberculosis, Lung Cancer, and Alzheimer's disease. The project integrates multiple machine learning models, robust user authentication, and a responsive web interface to provide an end-to-end diagnostic solution.

## Key Features

### User Authentication System
- *Registration & Login:* Secure user sign-up and sign-in.
- *Password Security:* Password hashing and recovery functionality.
- *Session Management:* Maintain user sessions for seamless navigation.
- *User Profiles:* Dashboard displaying user details and prediction history.

### Medical Image Analysis
- *Multi-Disease Prediction:* 
  - COVID-19 detection
  - Pneumonia detection
  - Tuberculosis (TB) detection
  - Lung Cancer detection
  - Alzheimer's disease detection
- *Image Preprocessing:* Each prediction model has its own preprocessing routine to resize, normalize, and support various image formats.

### Machine Learning Integration
- *TensorFlow Models:* 
  - COVID_Detect.keras
  - pneumonia_predict.keras
  - tb_model_final.keras
  - best_model_lung_cancer.keras
  - alzheimer_detect.keras
- *On-Demand Loading:* Models are loaded as needed, optimizing memory usage.
- *Prediction Outputs:* Provides confidence scores for each prediction, with results stored in the database.

### Database Management
- *SQLite & SQLAlchemy:* Manages user data, login activities, and historical prediction records.
- *Data Persistence:* Tracks user interactions and stores prediction history for future reference.

### Frontend Features
- *Responsive Design:* Modern UI that works seamlessly across devices.
- *Interactive Forms:* Easy image uploads and real-time feedback.
- *Dashboard & Profile Management:* View and manage prediction history and personal information.
- *Additional Pages:* Informative About page detailing the project.

### Technical Stack
- *Backend:* Flask 2.0.1
- *Database:* SQLite managed with SQLAlchemy
- *Authentication:* Flask-Login for secure access control
- *Machine Learning:* TensorFlow 2.8.0
- *Image Processing:* OpenCV and Pillow
- *Frontend:* HTML, CSS, JavaScript
- *Utilities:* NumPy, email-validator, and others

### Security Features
- *Password Hashing & CSRF Protection:* Ensuring secure user interactions.
- *Session Management & Protected Routes:* Prevent unauthorized access.
- *Secure File Handling:* Safeguarding image uploads and model files.

## Notable Implementation Details

### Image Processing Pipeline
- *Custom Preprocessing:* Unique functions for each medical condition ensure images meet model requirements.
- *Format Support:* Handles various image formats, resizing and normalization included.

### Prediction System
- *On-Demand Model Loading:* Efficient memory use by loading models as needed.
- *Result Storage:* Prediction outcomes, including confidence scores, are saved to the database.
- *User History:* Enables users to track and review their past predictions.

### User Experience
- *Interactive and Validated Forms:* Improves data input reliability and user satisfaction.
- *Responsive Design & Animations:* Utilizes modern UI elements and animations (powered by vanta.js) for an engaging experience.
- *Error Handling:* Provides clear feedback in case of issues.

### Data Management
- *User Profiles:* Manage personal data and view prediction history.
- *Activity Tracking:* Logs user login times and interactions.

## Project Structure

plaintext
mywebsite/
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
├── templates/          # HTML templates for the web pages
├── static/             # Static assets (CSS, JavaScript, images)
├── instance/           # Instance-specific files and configurations
└── *.keras            # Pre-trained machine learning model files

## Installation
- Follow these steps to set up the project locally:
  
  1. Clone the Repository:
  bash
  git clone https://github.com/shrrynsh/InVisionDX.git
  cd yourrepository
  
  2. Create a virtual environment:
  bash
  python3 -m venv venv
  source venv/bin/activate
  
  3. Install Dependencies:
  bash
  pip install -r requirements.txt
  
  4. Make sure that the file paths of models in app.py are correct.
## Usage
- Start the Application:
  bash
  python3 app.py
  
- Access the Application: Open your browser and navigate to http://127.0.0.1:5500.
- Register/Login:
  Create a new account or log in with existing credentials.
  Use your dashboard to upload images and view predictions.

- Making Predictions:
  Upload a medical image using the interactive form.
  View the prediction results along with confidence scores.
  Check your prediction history in your user profile.


![login](https://github.com/user-attachments/assets/610d15a1-5d39-4b86-a880-0936db3330cf)
![register](https://github.com/user-attachments/assets/6f452ff5-95e5-42b4-aeab-d3a2ad628f64)
![profile](https://github.com/user-attachments/assets/97ee35e0-ac49-4f2e-9183-94211e01491c)
![scan](https://github.com/user-attachments/assets/c4be4ef6-a9bd-409c-9896-5c85b74f60ea)
![project_success](https://github.com/user-attachments/assets/1c6554c0-b734-42fe-9c78-e3c24913943e)
