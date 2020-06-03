# Multimodal Analytics for Real-world News using Measures of Cross-modal Entity Consistency
This is the official GitHub page for the paper:

> Eric Müller-Budack, Jonas Theiner, Sebastian Diering, Maximilian Idahl, Ralph Ewerth:
"Multimodal Analytics for Real-world News using Measures of Cross-modal Entity Consistency".
Accepted for publication in: *International Conference on Multimedia Retrieval (ICMR)*, Dublin, 2020.

## News

**3rd June 2020:** We uploaded a **pre-release** of the *TamperedNews* and *News400* dataset with links to news texts, 
news images, and reference images. In addition, we provide the splits for validation and testing including the sets of 
untampered and tampered entities for each document. We are currently working on a script to download the news texts and 
images as well as the source code to reproduce our results. 

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

## LICENSE

This work is published under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. For details please check the
LICENSE file in the repository.