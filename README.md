
# Smart Waste Classification Using MobileNetV2

A Deep Learning and Computer Vision project that uses Transfer Learning to classify waste images into six different categories.

## About the Project

Waste management and proper waste segregation are important for environmental sustainability. This project aims to automate waste identification using Deep Learning.

The system takes an image of waste as input and predicts its category using a Convolutional Neural Network (CNN) based on the MobileNetV2 architecture.

## Waste Categories

The model is designed to classify waste into six categories:

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash

## Technologies Used

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras
- **Model Architecture:** MobileNetV2
- **Technique:** Transfer Learning
- **Frontend:** Streamlit
- **Image Processing:** Pillow
- **Data Processing:** NumPy
- **Visualization:** Matplotlib
- **Model Evaluation:** Scikit-learn
- **Development Environment:** VS Code

## Dataset

This project uses the TrashNet dataset.

Dataset Link: [TrashNet - Kaggle](https://www.kaggle.com/datasets/feyzazkefe/trashnet)

The dataset contains images belonging to six waste categories.

## Project Workflow

1. Dataset collection and preparation
2. Image preprocessing and augmentation
3. Building the MobileNetV2 model
4. Training using Transfer Learning
5. Fine-tuning the model
6. Model evaluation
7. Developing a Streamlit web application

## Project Structure

```text
smart-waste-classification-mobilenetv2/
│
├── app/                # Streamlit application
├── data/               # Dataset
├── models/             # Trained models
├── notebooks/          # Experiments and analysis
├── src/                # Training and preprocessing scripts
├── .gitignore
└── README.md
```

## Expected Output

The application will allow users to upload an image of waste and receive:

- Predicted waste category
- Prediction confidence score

## Learning Objectives

- Understand Computer Vision fundamentals.
- Learn Convolutional Neural Networks.
- Implement Transfer Learning.
- Work with pretrained MobileNetV2.
- Perform image preprocessing and augmentation.
- Evaluate a Deep Learning classification model.
- Build a web application using Streamlit.

## Project Status

🚧 Currently under development.

## Author

**Akula Rohith Kumar**

GitHub: [@rohithkumarakula](https://github.com/rohithkumarakula)
