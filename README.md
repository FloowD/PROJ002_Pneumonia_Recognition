# Pneumonia Recognition

Une implémentation Pytorch d'une détection de pneumonie en se basant sur des radios.


Création d'un modèle de détection de pneumonie en utilisant **Pytorch**.

En se basant principalement sur le document suivant : [Pneumonia Recognition](https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy/notebook)


## Requirements

```py
numpy==1.23.5
pandas==1.5.3
Pillow==9.4.0
tqdm==4.66.1
matplotlib==3.7.1
scikit-learn==1.2.2
seaborn==0.13.1
torch==2.1.0+cu121
tensorboard==2.15.1
mlxtend==0.22.0
```

## Pre-trained models
Les modèles pré-entrainées sont dans le dossier models

## Usage

Choissez un modèls dans le dossier `model` et une image dans le dossier `images`
```sh
python main.py --input_image INPUT_IMAGE --model MODEL
```

## Datasets

On a utilisé le dataset  [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) pour l'entrainement et le test

## Training

L'entrainement se fait dans le fichier main.ipynb

Vous pouvez également y avoir accès depuis le [google colab](https://colab.research.google.com/drive/1UgSolgs2_rvoCECqeDeSnp3VTQ728kpm?usp=sharing).