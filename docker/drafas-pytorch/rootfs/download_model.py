import torch
from torchvision import models
import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# Pre-download ResNet152
model = models.resnet152(pretrained=True)  # Download the model

# Save the model locally
save_path = f"{BASE_DIR}/resnet152_weights.pth"
torch.save(model.state_dict(), save_path)
print(f"ResNet152 model weights saved to {save_path}")
