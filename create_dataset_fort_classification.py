import random
from PIL import Image
import SimpleITK as sitk
import numpy as np
from datasets import Dataset
from tqdm import tqdm
import os

filenames = os.listdir('acouslic-ai-train-set/images/stacked_fetal_ultrasound')
all_examples = {
    0: [],
    1: [],
    2: [],
}

for filename in tqdm(filenames):
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

    # Examples qui correspondent à tuple(image, masque)
    examples = list(zip(images_array, masks_array))

    # Examples qui correspondent à {'label': 0, 1, ou 2, 'image': Image, 'annotation': Mask}
    examples = [add_label_to_example(example) for example in examples]

    for example in examples:
        if example['label'] == 0:
            if random.randint(1, 10) == 10:
                all_examples[example['label']].append(example)
        else:
            all_examples[example['label']].append(example)

# On convertit tout ça en dataset
# dataset = Dataset.from_list(all_examples)
# dataset.push_to_hub(repo_id='acouslic_ai', token='***')
max_number_of_elements = len(all_examples[1])
all_examples_to_be_uploaded = []

random.shuffle(all_examples[0])
random.shuffle(all_examples[1])
random.shuffle(all_examples[2])

all_examples_to_be_uploaded.extend(all_examples[0][:max_number_of_elements])
all_examples_to_be_uploaded.extend(all_examples[1][:max_number_of_elements])
all_examples_to_be_uploaded.extend(all_examples[2][:max_number_of_elements])

# On convertit tout ça en dataset
dataset = Dataset.from_list(all_examples_to_be_uploaded)
dataset.push_to_hub(repo_id='acouslic_ai_classification', token='***')