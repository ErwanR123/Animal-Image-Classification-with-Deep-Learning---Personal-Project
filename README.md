# Animal-Image-Classification-with-Deep-Learning---Personal-Project

This project showcases the development of a deep learning pipeline for animal image classification. Using a pre-trained ResNet model, it detects and classifies animal images into multiple categories with high accuracy. The project also includes a user-friendly web interface built with Streamlit for easy interaction.

---

## **Project Overview**

This project implements a full workflow for animal image classification, including:
- **Preprocessing**: Preparing and augmenting animal image datasets.
- **Training**: Using a ResNet-50 model pre-trained on ImageNet to fine-tune for animal classification.
- **Deployment**: Providing a Streamlit-based web app where users can upload images and get classification predictions.

---

## **Features**
1. **Image Preprocessing**:
   - Resize and normalize images for model compatibility.
   - Convert image labels into a machine-readable format.
2. **Model Training**:
   - Train a ResNet-50 model using PyTorch for multi-class classification.
   - Fine-tune the last fully connected layer to accommodate the dataset classes.
3. **Web Application**:
   - Upload an image and receive predictions along with confidence levels.
   - Visualize class probabilities using a bar chart.

---

## **Technologies Used**
- **Programming Language**: Python 
- **Libraries**:
  - `PyTorch` and `Torchvision`: For model training and image transformation.
  - `Streamlit`: For building the interactive web app.
  - `Pandas` and `NumPy`: For data manipulation.
  - `Matplotlib`: For visualizing prediction probabilities.
  - `Pillow`: For image loading and processing.

---

## **File Structure**
- **`app.py`**:
  - The Streamlit web application for user interaction and image classification.
- **`image_preprocessing.py`**:
  - Functions for loading, transforming, and preprocessing images.
- **`train_model.py`**:
  - Code for fine-tuning the ResNet-50 model on the animal dataset.
- **`trained_model.h5`**:
  - The saved model weights after training.

---

## **Dataset**
The dataset is sourced from Kaggle, providing high-quality images for training and testing. It is organized into folders representing each class (e.g., "Lion," "Tiger," "Elephant"), with corresponding images stored in their respective directories. Preprocessing ensures all images are resized, normalized, and ready for training.

---

## **Results**
1. **Accuracy**: The ResNet-50 model achieves high classification accuracy for animal images.
2. **Visualization**:
   - Displays top predictions with confidence percentages.
   - Shows a bar chart of probabilities for all classes.

---

## **Future Enhancements**
- Improving Model Accuracy: Actively working on a new version of the classification model to enhance accuracy and performance.
- Dataset Improvements: Refining the dataset to improve model training and generalization.

---

## **Acknowledgments**
Special thanks to the PyTorch and Streamlit communities for their comprehensive documentation and support.
The dataset used in this project was sourced from Kaggle, which provided high-quality images for training and testing the animal classification model.
