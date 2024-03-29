# Multimodal Analytics for Real-world News using Measures of Cross-modal Entity Consistency

This is the official GitHub page for the papers:

> Eric Müller-Budack, Jonas Theiner, Sebastian Diering, Maximilian Idahl und Ralph Ewerth. „Multimodal Analytics for Real-world News using Measures of Cross-modal Entity Consistency“. In: _International Conference on Multimedia Retrieval_, ICMR 2020, Dublin, Ireland, June 8-11, 2020. ACM, 2020, S. 16–25. DOI: https://doi.org/10.1145/3372278.3390670

> Eric Müller-Budack, Jonas Theiner, Sebastian Diering, Maximilian Idahl, Sherzod Hakimov und Ralph Ewerth. „Multimodal news analytics using measures of cross-modal entity and context consistency“. In: _International Journal of Multimedia Information Retrieval_ 10.2 (2021), Springer, S. 111–125. DOI: https://doi.org/10.1007/s13735-021-00207-4

## Supplemental Material

You can find the supplemental material here:
[supplemental_material](https://github.com/TIBHannover/cross-modal_entity_consistency/tree/master/supplemental_material)

## News

### 3rd June 2020:

- **Pre-release** of the _TamperedNews_ and _News400_ dataset with links to news texts, news images, untampered and
  tampered entity sets, and reference images for all entities.
- **Splits** for validation and testing
- **Download script** to crawl news texts

### 4th June 2020:

- **Full release** of the [_TamperedNews_](https://doi.org/10.25835/0002244) and
  [_News400_](https://doi.org/10.25835/0084897) dataset including the visual and textual features used in the
  paper
- **Inference scripts** and config files including the parameters used in the paper to reproduce the results for context
  and entity verification.

### 5th June 2020:

- **Download script** that automatically generates the whole dataset with the intended project structure
- **Docker container**
- Source code for **textual feature extraction**

### 22nd June 2020:

- **Image crawler** to obtain the news and reference images.

### 24th June 2020:

- Source code for **visual feature extraction**

### 13th January 2021:

- Added instructions to run docker with GPU support

### 16th July 2021:

- Added crawler to download reference images from bing
- Added functions for named entity recognition and linking
- Added inference script and examples ([Link](#Inference))

### 17th May 2022

- Release of the [updated version of both datasets _TamperedNews_ and _News400_](https://doi.org/10.25835/lzcs481w)
- Updated code including the [_VisE_ event classification approach](https://doi.org/10.1109/WACV48630.2021.00297) according to the [IJMIR'21 paper](https://doi.org/10.1007/s13735-021-00207-4)

## Content

This repository contains links to the _TamperedNews_ ([Link](https://doi.org/10.25835/0002244)) and
_News400_ ([Link](https://doi.org/10.25835/0084897)) datasets. Both datasets include:

## Content

For both datasets _TamperedNews_ and _News400_, we provide the:

- **`<dataset>.tar.gz`** containing the **`<dataset>.jsonl`** with:
  - Web links to the news texts
  - Web links to the news image
  - Outputs of the named entity recognition and disambiguation (NERD) approach
  - Untampered and tampered entities for each document
- **`<dataset>\_features.tar.gz`** with visual features for events, locations, and persons
- **`<dataset>\_wordembeddings.tar.gz`**: Word embeddings of all nouns in the news texts

For all entities detected in both datasets, we provide:

- **`entities.tar.gz`** containing an **`<entity_type>.jsonl`** for all entity types (events, locations, and persons) with:
  - Wikidata ID
  - Wikidata label
  - Meta information used for tampering
  - Web links to all reference images crawled from Google, Bing, and Wikidata
- **`entities_features.tar.gz`** containing the visual features of the reference images for all entities

The datasets can also be found here: https://doi.org/10.25835/0084897

We also provide source code and config files to reproduce our results:

- [eval_benchmark.py](https://github.com/TIBHannover/cross-modal_entity_consistency/blob/master/eval_benchmark.py)
  to reproduce the paper results
- [Config files](https://github.com/TIBHannover/cross-modal_entity_consistency/blob/master/test_yml) including the
  parameters used for the experiments in the paper

## Installation

We have provided a Docker container to execute our code. You can build the container with:

```shell script
docker build <PATH/TO/REPOSITORY> -t <DOCKER_NAME>
```

To run the container please use:

```shell script
docker run \
  --volume <PATH/TO/REPOSITORY>:/src \
  -u $(id -u):$(id -g) \
  -it <DOCKER_NAME> bash

cd /src
```

Add the flag `--gpus all` to the `docker run` command to run the code on your GPUs. For detailed instructions please follow: https://wiki.archlinux.org/index.php/Docker#Run_GPU_accelerated_Docker_containers_with_NVIDIA_GPUs

## Inference

Please download (and unpack) the models for the utilized deep learning models from the following links and place them in the respective directories of the project:

- `resources`
  - `event_classification` [model download](https://data.uni-hannover.de/dataset/3afb333d-230f-4829-91bb-d4dd41bfdcfa/resource/b105c1aa-3bc4-4233-8103-8f4616948d85/download/vise_co_cos.tar.gz)
    - `models`
      - `VisE_CO_cos.pt`
      - `VisE_CO_cos.yml`
    - `VisE-O_refined.graphml`
  - `facenet` [model download](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)
    - `20180402-114759.pb`
    - `model-20180402-114759.ckpt-275.data-00000-of-00001`
    - `model-20180402-114759.ckpt-275.index`
    - `model-20180402-114759.meta`
  - `geolocation_estimation` [model download](https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/base_M.tar.gz)
    - `cfg.json`
    - `model.ckpt.data-00000-of-00001`
    - `model.ckpt.index`
    - `model.ckpt.meta`
  - `scene_classification` [model download](http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar)
    - `resnet50_places365.pth.tar`

Please run the following command to apply the approach to a self-defined image-text pair:

```shell script
python infer.py \
  --config <PATH/TO/config.yml> \
  --text <PATH/TO/textfile.txt> \
  --image <PATH/TO/imagefile.jpg> \
  --wikifier_key <YOUR_WIKIFIER_API_KEY>
```

A Wikifier API key can be obtained by registering at http://www.wikifier.org/register.html.

You can specify the language with: `--language [en, de]` (`en` is default)

Two examples for testing and configs for the ICMR'20 and [IJMIR'21](https://doi.org/10.1007/s13735-021-00207-4) publications are provided in [examples](https://github.com/TIBHannover/cross-modal_entity_consistency/blob/master/examples)

An example on how to run the code can be found below:

```shell script
python infer.py \
  --config examples/config_ijmir21.yml \
  --text examples/Second_inauguration_of_Barack_Obama.txt \
  --image examples/Second_inauguration_of_Barack_Obama.jpg \
  --wikifier_key <YOUR_WIKIFIER_API_KEY>
```

We recommend using the config `examples/config_ijmir21.yml` of our latest approach presented in [IJMIR'21](https://doi.org/10.1007/s13735-021-00207-4).

Please note that the [icrawler](https://pypi.org/project/icrawler/) used to download reference images from the web, does not currently support crawling via _Google Images_. Instead 20 reference images are retrieved from _Bing_.

## Build Dataset

You can use the script provided in this repository to download and build the dataset:

```shell script
python build_dataset.py
```

This will automatically create a folder `resources` in the project containing the required data to execute the
following steps.

## Reproduce Paper Results

We have provided all necessary meta information and features to reproduce the results reported in the paper. This step requires to download all the dataset as described in [Build Dataset](#Build-Dataset). In case you have modified the
dataset paths, please specify the correct paths to the features, splits, etc. in the corresponding
[config files](https://github.com/TIBHannover/cross-modal_entity_consistency/blob/master/test_yml).

To reproduce the paper results, please run:

```shell script
python eval_benchmark.py --config test_yml/<config>.yml
```

or use the scripts provided in the [experiments](https://github.com/TIBHannover/cross-modal_entity_consistency/blob/master/experiments) folder

The number of parallel threads can be defined with: `--threads <#THREADS>`

## Build your own Models

We provide code to download news texts, images, and reference images to allow building your own system based on our datasets. In addition, the source code to extract textual and visual features used in our paper is provided.

### Download News Texts

The following command automatically downloads the text of the news articles:

```shell script
python download_news_text.py \
  --input <PATH/TO/dataset.jsonl>
  --output <PATH/TO/OUTPUT/DIRECTORY>
  --dataset <DATASET=[TamperedNews, News400]>
```

**Additional parameters:** Run the script with `--debug` to enable debugging console outputs.
The number of parallel threads can be defined with: `--threads <#THREADS>`

**Outputs:** This step stores a variety of meta information for each article with an ID `document_ID` in a file:
`<document_ID>.json`. In addition, the news texts are stored along with all other dataset information in a new
file: `dataset_with_text.jsonl`.

**Tip:** The script checks whether an article has already been crawled. We recommend running the script several times
as some documents might be missing due to timeouts in earlier iterations.

**Known Issues:** We are aware that some Websites have changed the news content or their overall template. For this reason, the texts can differ from our dataset. Please contact us (eric.mueller@tib.eu) for further information.

### Download Images

The following command automatically downloads the images of news articles or reference images for the entities found
in the dataset:

```shell script
python download_images.py \
  --input <PATH/TO/INPUT.jsonl> \
  --output <PATH/TO/OUTPUT/DIRECTORY> \
  --type <TYPE=[news, entity]>
```

**Additional parameters:**

Run the script with `--debug` to enable debugging console outputs.
You can set the dimension of the smaller image dimension to a maximal size using `--size <SIZE>`.
The number of parallel threads can be defined with: `--threads <#THREADS>`

To download the news images provide the path to the `dataset.jsonl` and run the script with `--type news`.

To download the references images of the entities found in the dataset, please provide the path to the respective
`<entity_type>.jsonl` and run the script with `--type entity`

### Extraction of Textual Features

If you haven't executed `build_dataset.py`, it is required to download the _fastText_ models for
[English](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz) or
[German](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz) for _TamperedNews_ and _News400_,
respectively. Put both models in the same folder `fasttext_folder`. The standard folder is `resources/fasttext`

You can extract the textual features of the news text using:

```shell script
python calculate_word_embeddings.py \
  --dataset <PATH/TO/dataset_with_text.jsonl> \
  --fasttext <PATH/TO/fasttext_folder> \
  --output <PATH/TO/OUTPUTFILE.h5>
```

### Extraction of Visual Features

Please download (and unpack) the models as described in [Inference](#Inference)

You can extract the visual features of the images downloaded according to [Download Images](#Download-Images) using:

```shell script
python calculate_image_embeddings.py \
  --input <PATH/TO/INPUT.jsonl> \
  --directory <PATH/TO/DOWNLOAD/FOLDER> \
  --model <PATH/TO/MODEL/FOLDER \
  --type <TYPE=[news, entity]> \
  --output <PATH/TO/OUTPUTFILE.h5>
```

Please note, that the path provided with `--directory` needs to match the output directory specified
in [Download Images](#Download-Images).

To generate the scene probabilities for all 365 _Places2_ categories, set the flag `--logits`

**Additional parameters:**
Run the script with `--debug` to enable debugging console outputs. Set the flag `--cpu` to generate the embeddings using a CPU.

**Credits**: We thank all the original authors for their work. The corresponding GitHub repositories are linked here:

- https://github.com/CSAILVision/places365
- https://github.com/davidsandberg/facenet
- https://github.com/TIBHannover/GeoEstimation
- https://github.com/TIBHannover/VisE

## LICENSE

This work is published under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. For details please check the
LICENSE file in the repository.
