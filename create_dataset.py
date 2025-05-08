from PIL import Image
import SimpleITK as sitk
import numpy as np
from datasets import Dataset
from tqdm import tqdm
import os

filenames = os.listdir('acouslic-ai-train-set/images/stacked_fetal_ultrasound')
all_examples = []

for filename in tqdm(filenames[:30]):
    images = sitk.ReadImage(f"acouslic-ai-train-set/images/stacked_fetal_ultrasound/{filename}")
    masks = sitk.ReadImage(f"acouslic-ai-train-set/masks/stacked_fetal_abdomen/{filename}")

    images_array = sitk.GetArrayFromImage(images)
    masks_array = sitk.GetArrayFromImage(masks)

    label_dict = {
        0: 'no_annotation',
        1: 'optimal',
        2: 'suboptimal'
    }

    def add_label_to_example(example: tuple[np.ndarray, np.ndarray]):
        image, mask = example

        if mask.sum(-1).sum(-1) == 0:
            label = 0
        elif 1 in mask:
            label = 1
            mask *= 255
        elif 2 in mask:
            label = 2
            mask[mask == 2] = 255
        else:
            raise ValueError("On n'a pas ce qu'on cherche")

        return {'label': label, 'image': Image.fromarray(image), 'annotation': Image.fromarray(mask)}

    # Exemples qui correspondent à tuple(image, masque)
    examples = list(zip(images_array, masks_array))

    # Exemples qui correspondent à {'label': 0, 1, ou 2, 'image': Image, 'annotation': Mask}
    examples = [add_label_to_example(example) for example in examples]

    all_examples.extend(examples)

# On convertit tout ça en dataset
dataset = Dataset.from_list(all_examples)
dataset.push_to_hub(repo_id='acouslic_ai', token='***')