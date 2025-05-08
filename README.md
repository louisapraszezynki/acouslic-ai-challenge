# ACOUSLIC-AI Challenge — Mesure automatique du périmètre abdominal fœtal
Ce repository contient le code et les notebooks utilisés pour le challenge ACOUSLIC-AI réalisé dans le cadre du projet de traitement d’image. Ce challenge d'imagerie biomédicale vise à développer des modèles d’intelligence artificielle capables de mesurer automatiquement le périmètre abdominal fœtal à partir de séquences échographiques 2D.

## Description des dossiers et des fichiers du repository
* `image_classification_model`  : notebook basé sur la documentation de Hugging Face pour l'entraînement du modèle de classification (ViT model)
* `image_segmentation_model`: notebook basé sur la documentation de Hugging Face pour l'entraînement du modèle de segmentation (SegFormer)
 
Scripts python développés pour le projet :
* `main.py` : premier script pour visualiser les données
* `mask.py` : affichage des masques
* `create_dataset.py` : création du dataset principal
* `create_dataset_for_classification.py` : préparation des données et création du dataset de classification
* `create_dataset_for_segmentation.py` : préparation des données et création du dataset de segmentation
* `test_our_trained_model_classification.py` : tests et inférence pour le modèle de classification
* `test_our_trained_model_segmentation.py` : tests et inférence pour le modèle de segmentation
* `validation.py` : inférence sur le set de validation avec nos 2 modèles

## Librairies principales utilisées
* `scikit-learn` (ConfusionMatrixDisplay)
* `matplotlib`
* `PIL`
* `SimpleITK`
* `numpy`
* `datasets` (Hugging Face)
* `transformers` (Hugging Face)

## Utilisation
Retrouver les datasets sur https://huggingface.co/Louloubib  
Création d'un compte wandb.ai  
Lancement des entraînements sur les notebooks (contient également du test/inférence mais sur des données de l'entraînement)  
Tests et inférence :  
* `python code/test_our_trained_model_classification.py`
* `python code/test_our_trained_model_segmentation.py`  
