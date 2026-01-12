import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
import joblib


# CONFIG
DATA_DIR = "DATA" #The colllection of DATA1,2,3,4
MODEL_PATH = "salafi_look_model.pkl"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# LOAD IMAGES
image_paths = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
]

train_paths, test_paths = train_test_split(
    image_paths, test_size=0.3, random_state=42
)


# FEATURE EXTRACTOR
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(DEVICE)
resnet.eval()

def extract_features(paths):
    feats = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            img = transform(img).unsqueeze(0).to(DEVICE)
            feat = resnet(img).cpu().numpy().flatten()
            feats.append(feat)
    return np.array(feats)


# TRAIN ONE-CLASS MODEL
X_train = extract_features(train_paths)

ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
ocsvm.fit(X_train)

joblib.dump(ocsvm, MODEL_PATH)
print("Model trained and saved.")


# TEST ACCEPTANCE RATE
X_test = extract_features(test_paths)
preds = ocsvm.predict(X_test)

acceptance = np.mean(preds == 1) * 100
print(f"Acceptance rate on test set: {acceptance:.2f}%")


# PREDICT ON NEW IMAGE
def predict_image(image_path):
    model = joblib.load(MODEL_PATH)

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = resnet(img).cpu().numpy().flatten().reshape(1, -1)

    pred = model.predict(feat)

    if pred[0] == 1:
        print("✅ SALAFI",image_path)
    else:
        print("❌ NOT SALAFI",image_path)


# EXAMPLE USAGE
# Replace this with the image you upload
predict_image("IMAGE")
