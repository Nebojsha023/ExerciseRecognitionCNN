import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, AlexNet_Weights
from PIL import Image
import os

# Proverimo da li racunar ima graficku
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Koristimo uredjaj: {device}")

# Definisemo model neuronske mreze
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

# Inicijalizacija modela
model, model_name = choose_model(num_classes=16)
model = model.to(device)

# Učitavanje modela
model_dir = 'models'
model_save_path = os.path.join(model_dir, model_name)

if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    print(f'Učitavanje modela iz {model_save_path}')
else:
    print(f'Nema postojeći model, trenirajte model i sačuvajte u {model_save_path}')

# Slika koju testiramo
lokacija_slike = 'C:\\Users\\Naache\\Desktop\\CNN\\example.jpg'  
image = Image.open(lokacija_slike).convert('RGB')

# Transformacija slike
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = transform(image)
image = image.unsqueeze(0).to(device)

model.eval()

with torch.no_grad():
    output = model(image)

_, predicted = torch.max(output, 1)
predicted_class = predicted.item()

naziv_vezbe = {
    0: 'bench', 1: 'squat', 2: 'deadlift', 3: 'barbell_curl', 
    4: 'chest_fly', 5: 'decline_bench', 6: 'hammer_curl', 
    7: 'hip_thrust', 8: 'incline_bench', 9: 'lat_pulldown', 
    10: 'lateral_raises', 11: 'leg_extension', 12: 'plank', 
    13: 'pull_up', 14: 'push_up', 15: 'shoulder_press'
}

vezba_na_slici = naziv_vezbe[predicted_class]
print(f'Vezba je : {vezba_na_slici}')