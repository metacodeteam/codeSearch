## Requirements

- Pytorch = 1.8.0
- python = 3.8.3
- dgl = 0.1.2
## Dataset

Our dataset is available in the "data" folder.

Dataset
Our dataset is available in the Google Drive(https://drive.google.com/drive/folders/1BTdsQNMwXMFEZ4bUHFVywnmhPTEjZ7-i?usp=sharing). To train the model with the dataset we created, you can copy the contents of the dataset folder to the /data folder of the model.

## Introduction

A:AST	C:CFG; 	P:PDG;	M:Mix(CFG+PDG);	T:Token

We show our models in "models" folder, and developers can choose the fusion strategy they want.

(Foe example, folder named "Fusion_TMA" is the usage of "Token+Mix+AST" fusion model, the details are in the folder.) 

## How to run

1. Configure a virtual environment and some packages

   ```
   pip install javalang
   pip install antlr4-python3-runtime
   ...
   ```


- choose the Fusion strategy(Fusion_TMA)

- set  the path to the data and other super parameters in config.py(super parameters)

- Run with the whole dataset,run:

  ```
  python train.py
  ```

- To evaluate a trained model, run:

  ```
  python test.py
  ```

  

