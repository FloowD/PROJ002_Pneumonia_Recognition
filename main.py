import argparse
import torch
from PIL import Image
from torchvision import transforms
from models import *


parser = argparse.ArgumentParser(description='Pneumonia X-Ray Recognition')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
args = parser.parse_args()

if not args.input_image or not args.model:
    print('You must provide an input image and a model file')
    parser.print_help()
    exit(1)

print(f'Input image: {args.input_image}')
print(f'Model file: {args.model}')

# Load the model
model = torch.load(args.model)
model.eval()

# Load the image
image = Image.open(args.input_image)
image = image.convert('RGB')

# Preprocess the image
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(500, 500)),
    transforms.Normalize([0.485, 0.56, 0.406], [0.229, 0.224, 0.225])
])
image = test_transform(image)
image = image.unsqueeze(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
image = image.to(device)

# Make a prediction
prediction = model(image)
if prediction < 0.5:
    print('Normal lungs')
else:
    print('Pneumonia detected')


