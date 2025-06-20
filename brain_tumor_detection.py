import os
import random
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Dataset class for brain tumor images
class BrainTumorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load and preprocess images
def load_images_from_folder(folder, label, img_size=64):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                img = img / 255.0
                images.append(img.astype(np.float32))
                labels.append(label)
    return images, labels

def load_dataset(base_path='brain_tumor_dataset', img_size=64):
    yes_folder = os.path.join(base_path, 'yes')
    no_folder = os.path.join(base_path, 'no')

    yes_images, yes_labels = load_images_from_folder(yes_folder, 1, img_size)
    no_images, no_labels = load_images_from_folder(no_folder, 0, img_size)

    X = np.array(yes_images + no_images)
    y = np.array(yes_labels + no_labels)

    return X, y

# Define CNN model in PyTorch
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Main training and evaluation function
def main():
    img_size = 64
    X, y = load_dataset(img_size=img_size)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create datasets and dataloaders
    train_dataset = BrainTumorDataset(X_train, y_train, transform=transform)
    test_dataset = BrainTumorDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    epochs = 5
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Save the trained model weights
    torch.save(model.state_dict(), "model.pth")
    print("Model weights saved to model.pth")

    # Evaluate model
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=["No Tumor", "Tumor"])
    print("\n📊 Classification Report:\n", report)

    # Simulate prediction and advice
    index = random.randint(0, len(X_test) - 1)
    sample_img = X_test[index]
    sample_tensor = transform(sample_img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(sample_tensor)
        _, predicted_class = torch.max(output, 1)
    print(f"\nPredicted: {'Tumor' if predicted_class.item() == 1 else 'No Tumor'}")
    if predicted_class.item() == 1:
        print("⚠️ Precautions:")
        print("- Consult a neurologist immediately.")
        print("- Avoid stress, take proper rest.")
        print("- Schedule MRI follow-ups.")
        print("- Follow prescribed treatment strictly.")
    else:
        print("✅ No tumor detected. Stay healthy!")

if __name__ == "__main__":
    main()
