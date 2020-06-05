# Multimodal Analytics for Real-world News using Measures of Cross-modal Entity Consistency
This is the official GitHub page for the paper:

> Eric Müller-Budack, Jonas Theiner, Sebastian Diering, Maximilian Idahl, Ralph Ewerth:
"Multimodal Analytics for Real-world News using Measures of Cross-modal Entity Consistency".
Accepted for publication in: *International Conference on Multimedia Retrieval (ICMR)*, Dublin, 2020.

## News

### 3rd June 2020: 

- **Pre-release** of the *TamperedNews* and *News400* dataset with links to news texts, news images, untampered and 
tampered entity sets, and reference images for all entities. 
- **Splits** for validation and testing
- **Download script** to crawl news texts

### 4th June 2020:
- **Full release** of the [*TamperedNews*]((https://doi.org/10.25835/0002244)) and 
[*News400*](https://doi.org/10.25835/0084897) dataset including the visual and textual features used in the 
paper
- **Inference scripts** and config files including the parameters used in the paper to reproduce the results for context 
and entity verification. 

### 5th June 2020:
- **Download script** that automatically generates the whole dataset with the intended project structure
- **Docker container**

### Future Releases
- **Web-crawler** to obtain the news and reference images.
- Source code for **visual and textual feature extraction**

## Content

This repository contains links to the *TamperedNews* ([Link](https://doi.org/10.25835/0002244)) and 
*News400* ([Link](https://doi.org/10.25835/0084897)) datasets. Both datasets include:

- **```<datasetname>```.tar.gz**:
    - ```dataset.jsonl``` containing:
        - Web links to the news texts
        - Web links to the news image
        - Outputs of the named entity recognition and disambiguation (NERD) approach
        - Untampered and tampered entities for each document
    - ```entity_type.jsonl``` file for each entity type containing the following information for each entity:
        - Wikidata ID
        - Wikidata label
        - Meta information used for tampering
        - Web links to all reference images crawled from Google, Bing, and Wikidata
    - splits for testing and validation
- **```<datasetname>```_features.tar.gz**:
    - Visual features of the news images for persons, locations, and scenes
    - Visual features of the reference images for persons, locations, and scenes
- **```<datasetname>```_wordembeddings.tar.gz**: Word embeddings of all nouns in the news texts

Based on the dataset we provide source code and config files to reproduce our results:

- [inference_context.py](https://github.com/TIBHannover/cross-modal_entity_consistency/blob/master/inference_context.py)
  to reproduce the results for context verification
- [inference_entities.py](https://github.com/TIBHannover/cross-modal_entity_consistency/blob/master/inference_entities.py)
  to reproduce the results for person, location, and event verification
- [Config files](https://github.com/TIBHannover/cross-modal_entity_consistency/blob/master/test_yml) including the
  parameters used for the experiments in the paper

## Usage

### Installation

We have provided a Docker container for easy installation. Please run the following command to build the container:
```shell script
docker build <PATH/TO/REPOSITORY> -t <DOCKER_NAME>
```

You can start all scripts in this repository using:
```shell script
docker run \
  --volume <PATH/TO/REPOSITORY>:/src \
  -u $(id -u):$(id -g) \
  -it <DOCKER_NAME> bash 
cd /src
```


### Build Dataset

Please download and build the dataset using the script provided in this repository. Simply run:
```shell script
python build_dataset.py
```
This will automatically create a folder ```resources``` in the project containing the required data to execute the 
following steps.

### Download News Texts

To download the text in the news articles run: 

```shell script
python download_news_text.py \
  -input <PATH/TO/dataset.jsonl> 
  -output <PATH/TO/OUTPUT/DIRECTORY> 
  -dataset <DATASET=[TamperedNews, News400]>
``` 

**Additional parameters:** Run the script with ```--debug``` to enable debugging console outputs.
The number of parallel threads can be defined with: ```--threads <#THREADS>```

**Outputs:** This step stores a variety of meta information for each article with an ID ```document_ID``` in a file: 
```<document_ID>.json```. In addition, the news texts are stored along with all other dataset information in a new 
file: ```dataset_with_text.jsonl```.

**Tip:** The script checks whether an article has already been crawled. We recommend to run the script several times 
as some documents might be missing due to timeouts.

### Cross-modal Entity Consistency

This step requires to download all features and word embeddings provided in the dataset. The features are stored in a 
folder called ```resources``` after running ```build_dataset.py```. In case you have modified the dataset paths, 
please specify the correct paths to the features, splits, etc. in the corresponding config files.

### Entity Verification

To reproduce the results for entity verification for a given entity type, please run:
```shell script
python inference_entities.py --config test_yml/<dataset_name>_<entity_type>.yml
```
The number of parallel threads can be defined with: ```--threads <#THREADS>```

### Context Verification

If you haven't executed ```build_dataset.py```, it is required to download the *fastText* models for 
[English](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz) or 
[German](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz) for *TamperedNews* and *News400*, 
respectively. Put both models in the same folder ```fasttext_folder```. The standard folder is ```resources/fasttext```

To reproduce the results for context verification, please run:
```shell script
python inference_context.py \
  --config test_yml/<dataset_name>_context.yml \
  --fasttext <PATH/TO/fasttext_folder>
```
The number of parallel threads can be defined with: ```--threads <#THREADS>```

## LICENSE

This work is published under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. For details please check the
LICENSE file in the repository.