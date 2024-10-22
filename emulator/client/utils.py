from PIL import Image
import numpy as np

# preprocessing image to fit input of resnet50 in triton
def preprocess_img(img: Image.Image, dtype, h, w):
    sample_img = img.convert('RGB')
    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    resized = resized.astype(dtype)

    return resized

