# Transfer Learning across Ontologies for Phenome-Genome Association Prediction

## Requirements

* Operating System - Linux 3.13.0
* MATLAB 8.1.0.604 (R2013a) or Octave 3.8.1

## How to run the project
You can run the project using the *script_main.m* file. This file will call the *runModel* function, which will perform cross-validation for parameter selection, followed by model evaluation by training with the best hyper-parameter obtained, and testing with test set. The options to be configurated are the following:

* **output_filepath**: path to the file which will contain the output;
* **evaluation_ontology**: ontology to be used in the evaluation step. Must be one of the following options: *"HPO"* or *"GO"*;
* **model**: the model to be run. Must be one of the following options: *"tlDLP"* (Transfer learning dual label propagation), *"DLP"* (dual label propagation), *"OGL"* (ontology-guided group Lasso), *"BiRW"* (Bi-Random Walk) or *"LP"* (label propagation);
* **gene_ontology**: the gene ontology to be used. Must be one of the following options: *"MF"* (molecular functions) or *"BP"* (biological processes);
* **train_model**: when *true*, the model will be trained with all the data, instead of performing cross-validation. Usually when this option is set, only one hyper-parameter option is set;
* **go_filepath**: path to the gene ontology file;
* **hpo_filepath**: path to the phenotype ontology file;
* **ppi_filepath**: path to the ppi-network file;
* **cv\_index_filepath**: path to the file containing the indices of cross-validation data (more information about CV data below);
* **tdlp\_Y0_filepath**: path to the initialization of tlDLP, usually the result from DLP or OGL (more information about tlDLP initiaization below);
* **fold_start** and **fold_end**: range of folds to use in the cross-validation. E.g.: If you want to perform 10-fold CV, *fold_start* must be set to 1, and *fold_end* to 10.

**Note:** Each model has a different number of hyper-parameters, also to be defined in this file;

## Phenotype ontology data

The phenotype ontology data can be found in the *data/phenotype_data* folder. Opening the file, you will find a data structure with the following fields:

* **phen_dag**: N x N matrix, in which N is the number of phenotypes. This matrix represents the associations between phenotypes;
* **phen_idx**: N x 1 vector, in which N is the number of phenotypes. Contains the HPO_ID for each phenotype;
* **depth**: N x 1 vector, in which N is the number of phenotypes. Contains the depth of each phenotype in the ontology, in which 1 is the root;
* **profiles**: M x N matrix, in which M is the number of genes and N is the number of phenotypes. Represents the associations between each gene and phenotype;
* **gene_table**: M x 1 vector, in which M is the number of genes. Each entry contains the gene name of the gene used by the profiles matrix.

## Gene ontology data

The gene ontology data can be found in the *data/gene_function_data* folder. Opening the file, you will find a data structure with the following fields:

* **phen_dag**: N x N matrix, in which N is the number of gene groups. This matrix represents the associations between gene groups;
* **phen_idx**: N x 1 vector, in which N is the number of gene groups. Contains the GO_ID for each gene group;
* **depth**: N x 1 vector, in which N is the number of gene groups. Contains the depth of each gene group in the ontology, in which 1 is the root;
* **profiles**: M x N matrix, in which M is the number of genes and N is the number of gene groups. Represents the associations between each gene and the groups;
* **gene_table**: M x 1 vector, in which M is the number of genes. Each entry contains the gene name of the gene used by the profiles matrix.

## PPI-Network

The PPI-network data can be found in the *data/ppi_data* folder. The file should contain one matrix of size N x N. N must be the number of genes in the intersection between the genes in the Gene Ontology and the Phenotype Ontology, considering only the genes with associations. 

## Cross-validation data

The cross-validation (CV) data can be found in the *data/CVindex* folder. The file contains a mc_CVindex cell, with K rows, where K is the number of folds. Each fold is a data structure containing the following fields:

* **trnIDX**: vector containing the indices of the associations in the training set, for model evaluation after hyper-parameters selector. If we are performing 5-fold CV, for example, this field should containg 80% of the indices;
* **tstIDX**: vector containing the indices of the associations in the test set, for model evaluation after hyper-parameters selector. If we are performing 5-fold CV, for example, this field should containg 20% of the indices;
* **CVtrnIDX**: vector containing the indices of the associations in the training set for hyper-parameters. If we are performing 5-fold CV, for example, this field should containg 80% of the indices in the training set (therefore 64% of the entire dataset). Notice that the union of *CVtrnIDX* and *CVtstIDX* is equal to *trnIDX*;
* **CVtstIDX**: vector containing the indices of the associations in the test set for hyper-parameters. If we are performing 5-fold CV, for example, this field should containg 20% of the indices in the training set (therefore 16% of the entire dataset). Notice that the union of *CVtrnIDX* and *CVtstIDX* is equal to *trnIDX*.

The indices in the file represent the indices of the associations between genes and the ontology. Let's say Y is M x N, such that M is the number of phenotypes and N the number of genes. By using `find(Y)` we receive the full list of associations between genes and phenotypes. The CV data contains the indices of this list.

## tlDLP initialization

In our tlDLP implementation the algorithm is initialized with the result of DLP (or OGL). Our analysis have shown better performance, comparing to initializing it with the training set. For this reason, the configuration of the *tdlp\_Y0_filepath* is required. This file should contain the Y matrix resulting from one of the other methods.

## Results

Results will be saved in the directory specified by the *output_filepath* variable. The file will contain a cell structure with K rows, in which K is the number of folds. Each fold contains the following:

* **datastr**: informs if was used molecular function of biological process gene ontology;
* **trnIDX**, **tstIDX**, **CVtrnIDX** and **CVtstIDX**: same as the cross-validation;
* **\<model_name>\_CVresult**: contains the number of associations correctly predicted in the top 100 rank, for each combination of hyper-parameters. This metric is used for hyper-parameters selection;
* **\<model_name>\_BParams**: contains the best hyper-parameters selected by cross-validation;
* **\<model_name>\_Yhat**: contains the matrix resulted from training the model with the best hyper-parameters, using the training-set;
