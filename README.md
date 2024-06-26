# Progeni

## Overview

This is the repository for the paper "A probabilistic knowledge graph for target identification." 
Progeni is a novel machine learning-based framework for target identification. 
Progeni integrates literature evidence with heterogeneous biological networks to construct a probabilistic knowledge graph, 
upon which graph neural networks are trained to learn the latent feature representations of targets, diseases, and various biological entities.
This repository contains all the data and code used throughout the paper. 

## Requirements

* pytorch (tested on version 1.10.0+cu102)
* numpy (test on version 1.22.0)
* sklearn (tested on version 1.0.1)
* scipy (tested on version 1.7.3)
* matplotlib (tested on version 3.5.1)
* argparse, rich(optional)

## Quick start

To reproduce our results:

1. Extract data.tgz, data_new.tgz, and literature_data.tgz at the project root folder
2. Go to the src/ folder
2. Run <code>generate_co.py</code> and <code>generate_co_new.py</code> to generate co-occurrence-dependent labels and weights.
3. Run <code>Progeni_cv.py</code> to reproduce cross validation results. Command line arguments are: 

```shell
--seed: global random seed. default: 26
--d: embedding dimension d. default: 1024
--n: global gradient norm to be clipped. default: 1
--k: dimension of project matrices k. default: 512
--model: model type, choices: Progeni_og, Progeni. default: Progeni
--l2-factor: weight of l2 regularization. default: 1
--lr: learning rate. default: 1e-3
--weight-decay: weight decay of optimizer. default: 0
--num-steps: number of training steps. default: 3000
--device: device number (-1 for cpu). default: 0
--n-folds: number of folds for cross validation. default: 5
--round: number of rounds of cross validation. default: 10
--test-size: portion of validation data w.r.t. trainval-set. default: .1
--mask: masking scheme, choices: random, tda_disease. default: random
```

The **entry-wise cross validation** corresponds to <code>--mask random</code> while the **cluster-wise cross validation** corresponds to <code>--mask tda_disease</code>.

3. Run <code>Progeni_retrain.py</code> to retrain the model on the full HN and save the model. Command line arguments are:

```shell
--seed: global random seed. default: 26
--d: embedding dimension d. default: 1024
--n: global gradient norm to be clipped. default: 1
--k: dimension of project matrices k. default: 512
--model: model type, choices: Progeni_og, Progeni. default: Progeni
--l2-factor: weight of l2 regularization. default: 1
--lr: learning rate. default: 1e-3
--weight-decay: weight decay of optimizer. default: 0
--num-steps: number of training steps. default: 3000
--device: device number (-1 for cpu). default: 0
--data: use old data (for computational analyses) or new data (for in-vitro validation)
```

## Data description

### Individual biological networks 

These data are in the data/ (data_new/) folders.

* drug.txt (drug.tsv): list of drug names.

* protein.txt (protein.tsv): list of protein names.

* disease.txt (disease.tsv): list of disease names.

* se.txt (se.tsv): list of side effect names.
* drug_dict_map: a complete ID mapping between drug names and DrugBank ID.
* protein_dict_map: a complete ID mapping between protein names and UniProt ID.
* mat_drug_se.txt : Drug-SideEffect association matrix.
* mat_protein_protein.txt : Protein-Protein interaction matrix.
* mat_drug_drug.txt : Drug-Drug interaction matrix.
* mat_protein_disease.txt : Protein-Disease association matrix.
* mat_drug_disease.txt : Drug-Disease association matrix.
* mat_protein_drug.txt : Protein-Drug interaction matrix.
* mat_drug_protein.txt : Drug-Protein interaction matrix.
* Similarity_Matrix_Drugs.txt : Drug & compound similarity scores based on chemical structures of drugs 
(for the file in data/, \[0,708) are drugs, the rest are compounds).
* Similarity_Matrix_Proteins.txt : Protein similarity scores based on primary sequences of proteins.

All entities (i.e., drugs, compounds, proteins, diseases and side-effects) are organized in the same order across all files. 
For the old data (in data/), these files: 
drug.txt, protein.txt, disease.txt, se.txt, drug_dict_map, protein_dict_map, mat_drug_se.txt, mat_protein_protein.txt, mat_drug_drug.txt, mat_protein_disease.txt, mat_drug_disease.txt, mat_protein_drug.txt, mat_drug_protein.txt, Similarity_Matrix_Proteins.txt, are extracted from https://github.com/luoyunan/DTINet.

### Co-occurrence counts

These data are in the literature_data/ folder

* protein_disease.txt (old) and prodis_new.txt (new): co-occurrence counts between proteins and diseases.
* protein_drug_new.txt (old) and prodrug_new.txt (new): co-occurrence counts between proteins and drugs.
* drug_disease_new.txt (old) and drugdis_new.txt (new): co-occurrence counts between drugs and diseases.

### Disease clusters

The clusters for cross-validation are pre-computed by data/cluster_data.py and stored in data/clus/disease_cluster.npy

### Output results

The prediction for the old dataset (output_raw_Progeni.npy) was uploaded to the topk_indices folder. 
The results for the new dataset can be downloaded from the following link: https://drive.google.com/file/d/1VHvs41f67_s5GvfqKVOfCKMZ76Zp-ZEb/view?usp=sharing.

## Contacts

If you have any questions or comments, please feel free to email Chang Liu (liu-chan19[at]mails[dot]tsinghua[dot]edu[dot]cn) and/or Jianyang Zeng (zengjy321[at]tsinghua[dot]edu[dot]cn).
