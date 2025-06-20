# 🧠 Brain Tumor Detection with PyTorch

Detect brain tumors from MRI images using a custom Convolutional Neural Network (CNN) built in PyTorch. The model classifies images into two categories: **Tumor** and **No Tumor**, and provides basic health precautions when a tumor is detected.

## 📌 Project Highlights

- 🧠 Detects brain tumors using grayscale MRI images
- ⚙️ Built from scratch using PyTorch (custom CNN)
- 📈 Reports accuracy, precision, recall, and F1-score
- 💡 Gives health advice if a tumor is found
- 💻 Runs on both CPU and GPU

## 🛠️ Tech Stack

- numpy
- opencv-python
- torch
- torchvision
- scikit-learn


## 📁 Project Structure

├── brain_tumor_detection.py         # 🧠 Main Python script for training, testing, and prediction
├── brain_tumor_dataset/             # 📂 MRI image dataset
│   ├── yes/                         # ✅ MRI images with tumors
│   └── no/                          # ❌ MRI images without tumors
├── model.pth                        # 💾 Saved trained model weights
├── requirements.txt                 # 📦 Required Python packages
└── README.md                        # 📘 Project documentation
     

## 📚 What I Learned

- Built a custom CNN model using PyTorch to classify brain MRI images into Tumor and No Tumor categories.
- Preprocessed real medical image data using OpenCV and PyTorch transforms (resizing, normalization).
- Trained and evaluated the model using metrics like accuracy, precision, recall, and F1-score.
- Implemented an advisory system that gives health precautions if a tumor is detected.
- Managed a full deep learning workflow — from loading data to saving model weights and predicting on unseen data.

  
