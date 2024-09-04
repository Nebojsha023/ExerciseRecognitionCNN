import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, AlexNet_Weights
from PIL import Image

import os
from tqdm import tqdm 

print('Pokretanje')

# Proverimo da li racunar ima graficku
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Koristimo uredjaj: {device}")

#Odabir modela
def choose_model(num_classes=16):
    print("Odaberite model za trening:")
    print("1: ResNet18")
    print("2: ResNet34")
    print("3: ResNet50")
    print("4: AlexNet")
    print("5: LeNet")

    choice = input("Unesite broj (1-5): ")

    if choice == '1':
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model_name = 'modelresnet18.pth'
    elif choice == '2':
        model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model_name = 'modelresnet34.pth'
    elif choice == '3':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model_name = 'modelresnet50.pth'
    elif choice == '4':
        model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        model_name = 'modelalexnet.pth'
    elif choice == '5':
        class LeNet(nn.Module):
            def __init__(self, num_classes=16):
                super(LeNet, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.conv2 = nn.Conv2d(6, 16, 5)
            
                self.fc1 = nn.Linear(16 * 53 * 53, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, num_classes)

            def forward(self, x):
                x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), (2, 2))
                x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
                x = x.view(-1, self.num_flat_features(x))
                x = torch.nn.functional.relu(self.fc1(x))
                x = torch.nn.functional.relu(self.fc2(x))
                x = self.fc3(x)
                return x

            def num_flat_features(self, x):
                size = x.size()[1:]  
                num_features = 1
                for s in size:
                    num_features *= s
                return num_features

        model = LeNet()
        model_name = 'modellenet.pth'
    else:
        raise ValueError("Broj mora biti izmedju 1 i 5.")

    return model, model_name

# Klasa za ucitavanje slika
class VezbeDataset(Dataset):
    def __init__(self, image_folder='C:\\Users\\Naache\\Desktop\\CNN\\data', transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = []
        self.labels = []

        self.naziv_vezbe = {
            'bench': 0, 'squat': 1, 'deadlift': 2, 'barbell_curl': 3, 
            'chest_fly': 4, 'decline_bench': 5, 'hammer_curl': 6, 
            'hip_thrust': 7, 'incline_bench': 8, 'lat_pulldown': 9, 
            'lateral_raises': 10, 'leg_extension': 11, 'plank': 12, 
            'pull_up': 13, 'push_up': 14, 'shoulder_press': 15
        }

        for naziv in os.listdir(image_folder):
            if naziv in self.naziv_vezbe:
                for image_file in os.listdir(os.path.join(image_folder, naziv)):
                    self.image_files.append(os.path.join(image_folder, naziv, image_file))
                    self.labels.append(self.naziv_vezbe[naziv])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        naziv = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, naziv

# Hiperparametri
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Transformacija podataka
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Ucitavanje dataseta
train_dataset = VezbeDataset(image_folder='C:\\Users\\Naache\\Desktop\\CNN\\data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Inicijalizacija modela
model, model_name = choose_model(num_classes=16)
model = model.to(device)

# Direktorijum u kojem cuvamo model
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
model_save_path = os.path.join(model_dir, model_name)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Progress bar
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train

    # Evaluation on the test set
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, '
          f'Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

print("Kreiranje modela gotovo.")

# Cuvanje modela
torch.save(model.state_dict(), model_save_path)
print(f'Model sačuvan u {model_save_path}')