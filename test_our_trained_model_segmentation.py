# Ici on va faire un script pour tester nos modèles sur toutes les images de notre dataset
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline

entire_dataset = load_dataset('Louloubib/acouslic_ai_filtered_for_segmentation')

print("Datasets loaded")

pipe = pipeline("image-segmentation", model="Louloubib/segformer_b0_acouslic_ai")

print("Pipeline loaded")


for sample in entire_dataset['train']:
    image = sample['image']
    original_mask = sample['annotation']
    out = pipe(image)
    generated_mask = out[0]['mask']

    original_mask.show()
    generated_mask.show()
    input()
