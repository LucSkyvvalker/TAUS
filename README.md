# Automated confidence scores for Machine Translations

This repository contains the code used in the Second Year project of the Bachelor Artificial Intelligence at the Univeristy of Amsterdam in collaboration with TAUS (https://www.taus.net/).

# Table of Contents
- [Contains](#Contains)
- [Basics](#Basics)
- [Reproduse](#Reproduse)
- [Portability](#Portability)
- [Dependencies](#Dependencies)

# Contains
- /models: directory containing zips of extracted features before splitting, and zips containing the ngram and ML models
- /scripts: directory containing all scripts used in the project

# Basics
- This project aims to create a confidence scoring alghorithm for machine translations. The confidence scores are given as probabilities, where 1 is a good translation and 0 a bad translation. All models found in this directory are not meant for actual use, as they have not been trained properly, nor are the PoS corpera suffienct for actual use.

# Reproduce
> Warning to use the .joblibs in this repository SKlearn v0.20.0 is needed.

- To reproduce our results the following things need to be done:
1. Install any dependencies and clone this repository
2. Create a directory /data and move the features.csv, ngram- and ML-models to this directory
3. Open a terminal in the scripts directory and run `$python3 train.py`, it will create a train and test set of the features.csv. If these files were allready made during a previous call of main.py it will ask if you wish to remake them.
4. You will then be prompted which model you wish to train. The options for this are:
  - `SVM`
  - `LR`
  - `MLP`
  - `NBC`
The training of these models may take some time, depending on the size of your dataset.
An in-depth explanation on each method and their results can be found in the accompanying report.
5. Run `$python3 scoreSentences.py`
It will prompt you for a model you wish to use, and continue to prompt for a "Source" and "Target" sentence. It will than give a confidence score for the target sentence to be correctly translated. To cancel the loop, press ctrl+c.
6. (Optional) Run `$python3 scoreModel.py`
It will prompt you for a model you wish to use and continue to give the F1-score, Accuracy  and mean Cross-Entropy of the model. 

# Portability
To use the scripts for training on new datsets the following steps must be taken:

> It is recommended to make back-ups of the /data folder before doing any of the following

 1. Place the dataset in the /data folder
 2.  Replace any calls to the dataset in the scripts.
 3. Run `python3 train.py` for each model you wish to train. 
 This cleans the dataset, extracts the features and creates train and test sets.

To use the scripts for training new N-Gram models:

 1. Remove all .joblibs from the /data folder.
 2. Place the desired corpera for the N-Gram models in the /data folder.
You will need 4 corpera in .txt format, a plain text and a POS tagged corpus for the source and target language.
3. Make sure that you have the SpaCy taggers for the correct languages loaded in `ngrams.py`
 4. Run `python3 train.py` 
The N-Gram models are trained automaticaly, after which you can retrain the models.

# Dependencies
- python 3.x
	- SKlearn 
	- pandas
	- numpy
	- SpaCy
		- SpaCy model: 'en_core_web_sm'
		- SpaCy model: 'nl_core_news_sm'
