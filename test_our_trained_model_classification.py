# Ici on va faire un script pour tester nos modèles sur toutes les images de notre dataset
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


entire_dataset = load_dataset("Louloubib/acouslic_ai_validation")
split_datasets = {
    0: entire_dataset["train"].filter(lambda e: e['label'] == 0),
    1: entire_dataset["train"].filter(lambda e: e['label'] == 1),
    2: entire_dataset["train"].filter(lambda e: e['label'] == 2),
}

print("Datasets loaded")

pipe = pipeline("image-classification", model="Louloubib/acouslic_ai_image_classification-10-epochs")

print("Pipeline loaded")

confusion_matrix = []

for dataset in [0, 1, 2]:
    results = {
        'suboptimal': 0,
        'optimal': 0,
        'no_annotation': 0,
    }


    for out in tqdm(
        pipe(KeyDataset(split_datasets[dataset], "image"), batch_size=64),
        desc=f"Iterating through dataset {dataset}",
        total=len(split_datasets[dataset])
    ):
        best_result = max(out, key=lambda e: e['score'])
        best_label = best_result['label']
        results[best_label] += 1

    print(f"Voici les résultats obtenus pour le dataset {dataset}")

    confusion_matrix_for_dataset = [
        float(results['no_annotation']) / (results['suboptimal'] + results['optimal'] + results['no_annotation']),
        float(results['optimal']) / (results['suboptimal'] + results['optimal'] + results['no_annotation']),
        float(results['suboptimal']) / (results['suboptimal'] + results['optimal'] + results['no_annotation'])
    ]

    confusion_matrix.append(confusion_matrix_for_dataset)

    print("Suboptimal:", float(results['suboptimal']) / (results['suboptimal'] + results['optimal'] + results['no_annotation']))
    print("Optimal:", float(results['optimal']) / (results['suboptimal'] + results['optimal'] + results['no_annotation']))
    print("No annotation:", float(results['no_annotation']) / (results['suboptimal'] + results['optimal'] + results['no_annotation']))
    print()

disp = ConfusionMatrixDisplay(confusion_matrix=np.array(confusion_matrix))
disp.plot()
plt.show()