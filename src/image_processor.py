import numpy as np
from PIL import Image
import os

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo {path} no existe.")
    try:
        img = Image.open(path)
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Error al cargar la imagen: {e}")

def compute_histogram(img_array, mode="rgb"):
    if len(img_array.shape) == 2:  # Escala de grises
        hist = np.zeros(256, dtype=int)
        for pixel in img_array.flatten():
            hist[pixel] += 1
        return hist
    else:  # Imagen a color (RGB)
        return [np.histogram(img_array[:, :, i], bins=256, range=(0, 255))[0] for i in range(3)]

def expand_histogram_custom(img_array, new_min=0, new_max=255):
    if len(img_array.shape) == 2:  # Escala de grises
        L, H = img_array.min(), img_array.max()
        if H == L:  # Evitar división por cero
            return np.full_like(img_array, new_min, dtype=np.uint8)
        expanded = ((img_array - L) / (H - L)) * (new_max - new_min) + new_min
        return np.clip(expanded, new_min, new_max).astype(np.uint8)
    else:  # Imagen a color (RGB)
        expanded = np.zeros_like(img_array)
        for i in range(3):
            L, H = img_array[:, :, i].min(), img_array[:, :, i].max()
            if H == L:
                expanded[:, :, i] = new_min
            else:
                expanded[:, :, i] = ((img_array[:, :, i] - L) / (H - L)) * (new_max - new_min) + new_min
        return np.clip(expanded, new_min, new_max).astype(np.uint8)
    
def equalize_histogram_custom(img_array, clip_limit=0.03):
    if len(img_array.shape) == 2:  
        hist = np.histogram(img_array, bins=256, range=(0, 255))[0]
        max_clip = int(np.sum(hist) * (1 - np.exp(-10 * clip_limit)))  
        hist = np.clip(hist, 0, max_clip)
        excess = np.sum(hist[hist == max_clip] - max_clip)
        hist = np.minimum(hist, max_clip) + excess // 256
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-10) 
        return cdf_normalized[img_array].astype(np.uint8)
    else:  
        equalized = np.zeros_like(img_array)
        for i in range(3):
            hist = np.histogram(img_array[:, :, i], bins=256, range=(0, 255))[0]
            max_clip = int(np.sum(hist) * (1 - np.exp(-10 * clip_limit)))  
            hist = np.clip(hist, 0, max_clip)
            excess = np.sum(hist[hist == max_clip] - max_clip)
            hist = np.minimum(hist, max_clip) + excess // 256
            cdf = hist.cumsum()
            cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-10)
            equalized[:, :, i] = cdf_normalized[img_array[:, :, i]]
        return equalized.astype(np.uint8)