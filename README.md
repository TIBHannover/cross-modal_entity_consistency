# Multimodal Analytics for Real-world News using Measures of Cross-modal Entity Consistency
This is the official GitHub page for the paper:

> Eric Müller-Budack, Jonas Theiner, Sebastian Diering, Maximilian Idahl, Ralph Ewerth:
"Multimodal Analytics for Real-world News using Measures of Cross-modal Entity Consistency".
Accepted for publication in: *International Conference on Multimedia Retrieval (ICMR)*, Dublin, 2020.

## News

**3rd June 2020:** We have uploaded a **pre-release** of the *TamperedNews* and *News400* dataset with links to news 
texts, news images, untampered and tampered entity sets, and reference images for all entities. In addition, we have 
provided the splits for validation and testing. 

**3rd June 2020:** Added download script for news texts. 

We are currently working on a script that automatically downloads the news and reference images. We will also provide 
the source code and data to reproduce our results. This includes:
- Functions to extract visual and textual features
- Extracted embeddings used in our paper
- Inference script including the evaluation metrics

## Content

This repository contains a **pre-release** of the 
*TamperedNews* ([Link](https://github.com/TIBHannover/cross-modal_entity_consistency/releases/download/0.1/tamperednews.tar.gz)) 
and *News400* ([Link](https://github.com/TIBHannover/cross-modal_entity_consistency/releases/download/0.1/news400.tar.gz)) 
dataset used in the paper. The datasets include:

- ```dataset.jsonl``` containing:
    - Web links to the news texts
    - Web links to the news image
    - Outputs of the named entity recognition and disambiguation (NERD) approach
    - Untampered and tampered entities
- ```<entity>.jsonl``` file for each entity type containing the following information for each entity:
    - Wikidata ID
    - Wikidata label
    - Meta information used for tampering
    - Web links to all reference images crawled from Google, Bing, and Wikidata
- splits for testing and validation
- ```download_news_text.py``` to download news texts from the urls provided in ```dataset.jsonl```

## Usage

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
```<document_id>.json```. In addition, the news texts are stored along with all other dataset information in a new 
file: ```dataset_with_text.jsonl```.

**Tip:** The script checks whether an article has already been crawled. We recommend to run the script several times 
as some documents might be missing due to timeouts.

## LICENSE

This work is published under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. For details please check the
LICENSE file in the repository.