# Projet Orchestration machine learning:
Sujet: Prédiction d'un temps de trajet à NYC

### Description du projet

Pour une entreprise fictive, on cherche à développer un pipeline ML entièrement automatisé qui répond à différents critères.

- Le pipeline ML doit être composé d'un **pipeline d'entraînement** et d'un **pipeline d'inférence**.
- Le modèle doit pouvoir être entraîné **automatiquement sur de nouvelles données**, et les équipes techniques doivent pouvoir auditer le comportement du modèle. Les données doivent avoir un minimum de **tendance saisonnière** afin de justifier l'entraînement régulier.

Stacks:

Kedro







# orchestration_machine_learning
1 - Création d'un environnement virtuel python :
    python -m venv env

2 -  Activation de l'environnement virtuel :
    Sous Windows : 
        env/Scripts/activate
    sous Linux:
        source env/bin/activate

3 - Installation du requirements :
    pip install -r requirements.txt

4 - Vérifier que Kedro a bien été installé :
    kedro info
<<<<<<< HEAD

5 - Création d'un projet vide kedro : 
    kedro new
=======
>>>>>>> c15ef77ecdcdf19a5ea16d24ca0bfce075952ef9
