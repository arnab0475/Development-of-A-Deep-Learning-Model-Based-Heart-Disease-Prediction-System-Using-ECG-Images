# Development-of-A-Deep-Learning-Model-Based-Heart-Disease-Prediction-System-Using-ECG-Images

This project demonstrates the use of Convolutional Neural Networks (CNNs) and Transfer Learning (VGG16) for classifying ECG (Electrocardiogram) images into different categories based on visual patterns. The model is trained and evaluated using the ECG_DATA dataset on Kaggle.

📁 Dataset Structure
bash
Copy
Edit
ECG_DATA/
├── train/
│   ├── ClassA/
│   ├── ClassB/
│   └── ...
└── test/
    ├── ClassA/
    ├── ClassB/
    └── ...
Each folder contains .png, .jpg, or .jpeg ECG image files.

🧠 Project Workflow
1. 📊 Data Exploration
Loaded training and test ECG image data.

Counted images in each class.

Displayed sample images from each class.

2. 🧹 Data Preprocessing
Rescaled images using ImageDataGenerator.

Created training, validation, and test generators with a target image size of (224, 224).

3. 🧱 Model Building
🔸 Custom CNN Model
Built a Sequential CNN model with Conv2D, MaxPooling2D, Flatten, and Dense layers.

Compiled using Adam optimizer and categorical crossentropy loss.

🔸 VGG16 Transfer Learning
Used pretrained VGG16 (without top layer) as base model.

Added custom classifier on top.

Froze base layers for feature extraction.

4. 🏋️ Model Training
Trained both models for 10 epochs.

Used validation set to monitor accuracy and loss.

5. 📈 Performance Evaluation
Evaluated models on the test set.

Generated:

Accuracy & Loss plots

Confusion Matrix

Classification Report

6. 🔍 Predictions & Visualization
Predicted and visualized:

Random samples

Class-wise examples

Confidence scores

📊 Results
Test Accuracy (VGG16-based model): Displayed in final evaluation

Plotted:

Training vs Validation Accuracy & Loss

Confusion Matrix using Seaborn

🗃️ Output
Saved trained model as: ecg_vgg16_model.h5

Used for future predictions on unseen ECG images.

🧪 Requirements
Python 3.x

TensorFlow / Keras

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

All libraries are pre-installed in Kaggle's Python environment.

🔮 Future Improvements
Data augmentation for better generalization.

Fine-tuning VGG16 or experimenting with other pretrained models (e.g., ResNet50).

Hyperparameter tuning (learning rate, batch size, etc.).

Deploy the model as a web app using Streamlit or Flask.

📷 Sample Predictions
The notebook includes real image predictions like:

✅ True Label: Normal

🤖 Predicted: Myocardial Infarction

📊 Confidence Score: 92.4%
