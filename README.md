# Quadruple Evaluation Network - PyTorch Implementation

This is the source code for Quadruple Evaluation Network for the task of taxonomy completion, published in The Web Conference 2022.

## Installation

The codebase was tested under python 3.8.10 and cuda 11.1.1.

### Installation with enviroment.yml

1. Clone this repository.

   ```
   git clone https://github.com/sheryc/QEN.git
   ```

2. Create a new conda environment with the ```environment.yml```:

   ```
   conda env create -n qen -f QEN/environment.yml
   ```

And it's ready to run!

### Manual Installation

For manual installation (e.g., on slurm clusters), please run the following commands:

```
# Load python 3.8 and cuda 11.1
pip install numpy networkx wandb gensim tqdm more_itertools transformers spacy pandas
pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m spacy download en_core_web_md
```

## Data Preparation

### For datasets used in the paper

We have provided the three datasets we used in the paper below. Put the corresponding extracted folder directly under `data/` for the experiments.

* [WordNet-Verb](https://drive.google.com/file/d/13NXEAsy4sBFzq4C_lSYAhiiOgFCoG7Qa/view?usp=sharing)
* [MeSH](https://drive.google.com/file/d/1hCxIdwoHbb11q9V9kZSXV3f-4lU9FQzr/view?usp=sharing)
* [SemEval-Food](https://drive.google.com/file/d/1tZbGp9ayWCMRnBWe38vCsrNmkO5Lu3-K/view?usp=sharing)

Each of the dataset folders contain at least the following files:

1. ``<TAXO_NAME>.terms``. Each line in this file represents one term / concept / node in the taxonomy, including its *ID* and *surface name*.

```
taxon1_id \t taxon1_surface_name
taxon2_id \t taxon2_surface_name
taxon3_id \t taxon3_surface_name
...
```

2. ``<TAXO_NAME>.taxo``. Each line in this file represents one relation / edge in the taxonomy, including the *parent taxon ID* and *child taxon ID*.

```
parent_taxon1_id \t child_taxon1_id
parent_taxon2_id \t child_taxon2_id
parent_taxon3_id \t child_taxon3_id
...
```

3. ``<TAXO_NAME>.desc``. Each line in this file represents the extracted term description of each term, mapped to each term's *surface name*.

```
taxon1_surface_name \t taxon1_description
taxon2_surface_name \t taxon2_description
taxon3_surface_name \t taxon3_description
...
```

4. ``<TAXO_NAME>.pickle.bin``. This file is created by ``dataloader/dataset`` when encountering a raw dataset. The pickled dataset contains taxonomy information as well as train/val/test splits.

### For your own taxonomy

* Step 1: Organize your input taxonomy along with node features into the format of ``<TAXO_NAME>.terms``, ``<TAXO_NAME>.taxo`` and ``<TAXO_NAME>.desc`` mentioned in the previous section.

  * Step 1.1: If the description file is unavailable, you can use the provided Wikipedia description generator in ``data/description.py`` to generate the description file.

   ```
   description.py -d <TAXO_NAME> -m wikipedia
   ```

* Step 2: (Optional) Generate the train/val/test splits in files called ``<TAXO_NAME>.terms.train``, ``<TAXO_NAME>.terms.validation`` and ``<TAXO_NAME>.terms.test``. The current format of these files should contain the *line number* of the corresponding terms in ``<TAXO_NAME>.terms`` file. We will improve this as soon as possible.

* Step 3: Run the training script by setting the argument `raw` for the class `MAGDataset` in `data_loader/dataset.py` to `True`. After training once, the script will generate the pickled dataset and `raw` can be set to `False` for future experiments.

## Model training

For reproducing the results in the paper, please use the provided configs in `config_files/<TAXO_NAME>/config.text.c.json`.

```
python train.py --config config_files/<TAXO_NAME>/config.test.c.json --exp <WANDB_RUN_TAG>
```

For running QEN for your own taxonomy, please create a config file similar to the ones provided in the corresponding folder, and run the experiments with the above script.


### Model Organization

For all implementations, we follow the project organization in [pytorch-template](https://github.com/victoresque/pytorch-template).

## Reference

Please cite our paper if this code contributes to an academic publication:

```
@inproceedings{wangQENApplicableTaxonomy2022,
author = {Wang, Suyuchen and Zhao, Ruihui and Zheng, Yefeng and Liu, Bang},
title = {QEN: Applicable Taxonomy Completion via Evaluating Full Taxonomic Relations},
year = {2022},
isbn = {9781450390965},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3485447.3511943},
booktitle = {Proceedings of the ACM Web Conference 2022},
pages = {1008–1017},
numpages = {10},
location = {Virtual Event, Lyon, France},
series = {WWW '22}
}
```
