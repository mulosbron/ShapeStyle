# ShapeStyle

## Overview
An AI-powered web application that analyzes user photos to determine face shapes and provides personalized hairstyle recommendations. Using deep learning and transfer learning techniques, the system classifies faces into five categories (Heart, Oblong, Oval, Round, Square) and suggests optimal hairstyles for each shape.

## Purpose
To help users find the most suitable hairstyle for their face shape through automated facial analysis. The application aims to accelerate decision-making in hairstyle selection and ensure users make the right choice by matching their facial features with predefined hairstyle recommendations.

## Scope

### Technology Stack:
- **Deep Learning**: VGGFace, Keras 2.2.4, TensorFlow 1.14.0
- **Face Detection**: MTCNN
- **Web Development**: Python Flask, HTML/CSS, JavaScript
- **Data Processing**: NumPy, OpenCV, joblib, pickle
- **Transfer Learning**: Pre-trained VGG16 architecture

## Implementation

### Project Structure:
```
ShapeStyle/
├── model/
│   ├── face_shape_model_vgg16_rgb.h5    # Trained model
│   ├── evaluation_results/               # Performance metrics and graphs
│   ├── input_imgs/                       # Test images
│   ├── output_results/                   # Prediction results
│   ├── face_shape_trainer.py                    # Model training script
│   ├── face_shape_predictor.py                     # Model testing script
│   └── face_shape_evaluator.py           # Model evaluation script
├── preprocessing/
│   ├── preprocessing.py                  # Data preprocessing script
│   └── preprocessing_control.py          # Data validation script
├── hair_style_recommender/
│   ├── backend/
│   │   └── api.py                       # Flask API server
│   └── frontend/
│       ├── face_shape_classifier.html   # Main interface
│       ├── recommended_styles.html      # Hairstyle recommendations
│       └── [face_shape_folders]/        # Hairstyle images by face shape
└── requirements.txt
```

### Model Performance:
- **Accuracy**: 89.60%
- **Precision (Weighted)**: 89.62%
- **Recall (Weighted)**: 89.60%
- **F1-Score (Weighted)**: 89.59%
- **AUC Score (Weighted)**: 98.84%

### Classification Report:
| Face Shape | Precision | Recall | F1-Score | Support |
|------------|-----------|---------|----------|---------|
| Heart      | 0.92      | 0.93    | 0.92     | 200     |
| Oblong     | 0.93      | 0.93    | 0.93     | 200     |
| Oval       | 0.89      | 0.84    | 0.87     | 200     |
| Round      | 0.86      | 0.89    | 0.87     | 200     |
| Square     | 0.89      | 0.90    | 0.89     | 200     |

## Screenshots

### System Workflow
<img src="hair_style_recommoender/1.jpg" width="400" alt="Step 1: Capture Photo">
<img src="hair_style_recommoender/2.jpg" width="400" alt="Step 2: Face Shape Prediction">
<img src="hair_style_recommoender/3.jpg" width="400" alt="Step 3: Hairstyle Recommendations">

### Model Performance Visualizations
<img src="model/evaluation_results/confusion_matrix_counts.png" width="400" alt="Confusion Matrix">
<img src="model/evaluation_results/roc_curves.png" width="400" alt="ROC Curves">

### Dataset:
- 5,000 celebrity images categorized by face shapes
- 1,000 images per category (Heart, Oblong, Oval, Round, Square)
- Training set: 800 images per category
- Test set: 200 images per category
