import numpy as np
from PIL import Image
import SimpleITK as sitk
from tqdm import tqdm
import os

from transformers import pipeline

image_classifier = pipeline("image-classification", model="Louloubib/acouslic_ai_image_classification-10-epochs")
image_segmenter = pipeline("image-segmentation", model="Louloubib/segformer_b0_acouslic_ai")

# Utilisation des 30 dernières images = set de validation
filenames = os.listdir('acouslic-ai-train-set/images/stacked_fetal_ultrasound')[270:]

# A partir des 840 frames d'une image, renvoie la frame "optimale"
# Utilise le modèle de classification (image_classifier)
def get_best_frame_from_mha(frames):
    images_array = sitk.GetArrayFromImage(frames)
    images_list = list(images_array)

    highest_optimal_score = 0
    kept_index = 0

    for i, image_array in enumerate(images_list):
        image = Image.fromarray(image_array)
        result = image_classifier(image)
        optimal_score = [r for r in result if r['label'] == 'optimal'][0]['score']

        if optimal_score > highest_optimal_score:
            highest_optimal_score = optimal_score
            kept_index = i

    return images_list[kept_index], highest_optimal_score

# A partir d'une frame, renvoie le masque
# Utilise le modèle de segmentation (image_segmenter)
def get_mask_from_frame(frame):
    mask = image_segmenter(frame)
    mask = [r for r in mask if r['label'] == 'annotation'][0]['mask']
    return np.array(mask)

# Inférence sur le set de validation (30 dernières images)
for filename in tqdm(filenames):
    images = sitk.ReadImage(f"acouslic-ai-train-set/images/stacked_fetal_ultrasound/{filename}")
    image, score = get_best_frame_from_mha(images)

    mask = get_mask_from_frame(Image.fromarray(image))
    image = np.stack([image, image, image], axis=-1)
    image[:, :, 1] += (mask / 10).astype(np.uint8)

    Image.fromarray(image).show()

