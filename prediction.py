from PIL import Image
import yaml
from torchvision import transforms, models


def load_model():
  net = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
  return net


def read_image(image_encoded):
  img = Image.open(str(image_encoded))
  return img


def preprocess(img):
  trans = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])
  x = trans(img)
  return x.unsqueeze(dim=0)


def predict(x):
  with open('./imagenet_labels.yaml', 'r') as f:
    labels = yaml.safe_load(f)
  
  net = load_model()
  if net is None:
    net = load_model()
  net.eval()
  
  output = net(x)
  _, y_pred = output.max(1)
  pred_name = labels[y_pred.item()]
  return pred_name[0]