# Automated confidence scores for Machine Translations

This repository contains the code used in the Second Year project of the Bachelor Artificial Intelligence at the Univeristy of Amsterdam in collaboration with TAUS (https://www.taus.net/).

# Table of Contents
- [Contains](#Contains)
- [Basics](#Basics)
- [Reproduse](#Reproduse)
- [New Models](#NewModels)
- [Dependencies](#Dependencies)

# Contains
- /models: directory containing zips of extracted features before splitting, and zips containing the ngram and ML models
- /scripts: directory containing all scripts used in the project

# Basics
- This project aims to create a quality estimator for machine translations. The quality estimations are given as a confidence score in the range \[0-1\]. All models found in this directory are not meant for actual use, as they have not been trained properly, nor are the PoS corpera suffienct for actual use.

# Reproduce
- To reproduce our results the following things need to be done:
1. Install any dependencies and clone this directory
2. Create a directory /data and move the features.csv, ngram- and ML-models to this directory
3. Open a terminal in the scripts directory and run `$python3 main.py`, it will create a train and test set of the features.csv with a 4:1 ratio. If these files were allready made during a previous call of main.py it will ask if you wish to remake them.
4. You will then be prompted with which Machine Learning method you wish to train based on the train and test sets. The options for this are(this may take some time, depending on the size of your train and test sets):
  - SVM
  - LR
  - MLP
  - NBC
An in-depth explanation on each method and their results can be found in the accompanying paper.
5. From the same terminal run `$python3 confidenceScore.py` it will prompt you for a Machine Learning model you wish to use, and continue to prompt for a "Source" and "Target" sentence. It will than give a confidence score for the target sentence to be correctly translated. To cancel the loop, press ctrl+c.
6. (Optional) From the same terminal run `$python3 getScores.py`, it will prompt you for a Machine Learning model you wish to use and continue to give the Fscore and cross-entropy of the model based on the test set created in step 3.

# NewModels
- To use the scripts to create new models, for example for different language pairs, better PoS corpera or more features, a new features.csv and new n-gram models must be trained. Empty the /data directory and put a new dataset into the folder. You may have to change the file 'cleandatas.py' and change the filenames in 'main.py'. Then proceed with step 3 from Reproduce(#Reproduce)

# Dependencies
- python 3.x
- pandas
- numpy
- SpaCy
