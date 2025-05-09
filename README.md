# ACOUSLIC-AI Challenge — Mesure automatique du périmètre abdominal fœtal
Ce repository contient le code et les notebooks utilisés pour le challenge ACOUSLIC-AI réalisé dans le cadre du projet de traitement d’image. Ce challenge d'imagerie biomédicale vise à développer des modèles d’intelligence artificielle capables de mesurer automatiquement le périmètre abdominal fœtal à partir de séquences échographiques 2D.

## Description des dossiers et des fichiers du repository
Notebooks Collab pour les entraînements des modèles, basés sur la documentation de Hugging Face :
* `image_classification_model`  : pour le modèle de classification (ViT model)
* `image_segmentation_model`: pour le modèle de segmentation (SegFormer)
 
Scripts python développés pour le projet :
* `create_dataset.py` : création du dataset principal
* `create_dataset_for_classification.py` : préparation des données et création du dataset de classification
* `create_dataset_for_segmentation.py` : préparation des données et création du dataset de segmentation
* `test_our_trained_model_classification.py` : tests et inférence pour le modèle de classification
* `test_our_trained_model_segmentation.py` : tests et inférence pour le modèle de segmentation
* `validation.py` : inférence sur le set de validation avec nos 2 modèles

## Utilisation
- Retrouver les datasets sur https://huggingface.co/Louloubib  
- Création d'un compte https://wandb.ai  
- Lancement des entraînements sur les notebooks sur https://colab.research.google.com/ (contient également du test/inférence mais sur des données de l'entraînement)  
- Tests et inférence de chaque modèle indépendemment sur `test_our_trained_model_classification.py` et `test_our_trained_model_segmentation.py`
- Test et inférence sur la méthode complète sur `validation.py`
  
## Exemples de résultats
![image](https://github.com/user-attachments/assets/da625242-d1cd-4242-ab4c-aaff30edd9dd)
Exemples de sélection de 3 frames optimales et leur masque pour la mesure du PA (en vert)

## Librairies principales utilisées
* `scikit-learn` (ConfusionMatrixDisplay)
* `matplotlib`
* `PIL`
* `SimpleITK`
* `numpy`
* `datasets` (Hugging Face)
* `transformers` (Hugging Face)

    
