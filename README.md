# ML-predictions-of-Antibody-Non-Specificity

This repository contains the code used in the manuscript "Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters". 

Original publication: https://doi.org/10.1101/2025.04.28.650927  
Zenodo (Harvey dataset): https://doi.org/10.5281/zenodo.17962178  

## Overview
The development of therapeutic antibodies requires optimizing target binding affinity and pharmacodynamics, while ensuring high developability potential, including minimizing non-specific binding. In this study, we address this problem by predicting antibody non-specificity by two complementary approaches:
- antibody sequence embeddings by protein language models (PLMs)
- a comprehensive set of sequence-based biophysical descriptors.

## Usage
1. Install environment:   
```
conda env create -f environment.yml
```  
3. Get the sequence embedding according to the `01_Embeddings.ipynb` notebook.
4. Load the model of interest and run inference:
```
with open('./Data/Models/model_esm1v_VH_LogisticReg.pkl', 'rb') as file:  
    model = pickle.load(file)
y_predict = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]
```
See `04_Validation.ipynb` notebook for further details and examples.

## Citation
If you use this code in your research, please cite: https://doi.org/10.1101/2025.04.28.650927

## Licence
This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
Please contact Laila Sakhnini llsh@novonordisk.com or Ludovica Beltrame lvbl@novonordisk.com to report issues of for any questions.
