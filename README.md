
PDF Trio
============

**Git Repo:** <http://github.com/internetarchive/pdf_trio>

**Blog Post:** <https://towardsdatascience.com/making-a-production-classifier-ensemble-2d87fbf0f486>

**License:** Apache-2.0

`pdf_trio` is a Machine Learning (ML) service which combines three distinct
classifiers to predict whether a PDF document is a "research publication". It
exposes an HTTP API which receives files (POST) and returns classification
confidence scores. There is also an experimental endpoint for fast
classification based on URL strings alone.

This project was developed at the [Internet Archive](https://archive.org) as
part of efforts to collect and provide perpetual access to research documents
on the world wide web. Initial funding was provided by the Andrew W. Mellon
Foundation under the "[Ensuring the Persistent Access of Long Tail Open Access
Journal Literature][mellon-blog]" project.

This system was originally designed, trained, and implemented by [Paul
Baclace](http://baclace.net/). See `CONTRIBUTORS.txt` for details.

[mellon-blog]: https://blog.archive.org/2018/03/05/andrew-w-mellon-foundation-awards-grant-to-the-internet-archive-for-long-tail-journal-preservation/


## Quickstart with Docker Compose

These instructions describe how to run the `pdf_trio` API service locally using
docker and pre-trained Tensorflow and fastText machine learning models. They
assume you have docker installed locally, as well as basic command line tools
(bash, wget, etc).

**Download trained model files:** about 1.6 GBytes of data to download from
archive.org.

    ./fetch_models.sh

**Run docker-compose:** this command will build a docker image for the API from
scratch and run it. It will also fetch and run two tensorflow-serving back-end
daemons. Requires several GByte of RAM.

    docker-compose up

You can try submitting a PDF for classification like:

    curl localhost:3939/classify/research-pub/all -F pdf_content=@tests/files/research/hidden-technical-debt-in-machine-learning-systems__2015.pdf

To re-build the API docker image (eg, if you make local code changes):

    docker-compose up --build --force-recreate --no-deps api

## Development 

The default python dependency management system for this project is `pipenv`,
though it is also possible to use `conda` (see directions later in this
document).

To install dependencies on a Debian buster Linux computer:

    sudo apt install -y poppler-utils imagemagick libmagickcore-6.q16-6-extra ghostscript netpbm gsfonts wget
    pip3 install pipenv
    pipenv install --dev --deploy

Download model files:

    ./fetch_models.sh

Use the default local configuration:

    cp example.env .env

Run just the tensorflow-serving back-end daemons using docker-compose like:

    docker-compose up tfserving

Unit tests partially mock the back-end tensorflow-serving daemons, and any
tests which do call these daemons will automatically skip if they are not
available locally. To run the tests:

    pipenv run python -m pytest
    pipenv run pylint -E pdf_trio tests/*.py

    # with coverage:
    pipenv run pytest --cov --cov-report html

## Background

The purpose of this project is to identify research works for richer cataloging
in production at [Internet Archive](https://archive.org). Research works are
not always found in well-curated locations with good metadata that can be
utilized to enrich indexing and search. [Ongoing
work](https://blog.dshr.org/2015/04/preserving-long-form-digital-humanities.html)
at the Internet Archive will use this classifier ensemble to curate "long tail"
research articles in multiple languages published by small publishers.  [Low
volume
publishing](https://blog.dshr.org/2017/01/the-long-tail-of-non-english-science.html)
is inversely correlated to longevity, so the goal is to preserve the research
works from these sites to ensure they are not lost.  

The performance target is to classify a PDF in about 1 second or less on
average and this implementation achieves that goal when multiple, parallel
requests are made on a multi-core machine without a GPU. 

The URL classifier is best used as a "true vs. unknown" choice, that is, if the
classification is non-research ('other'), then find out more, do not assume it
is not a research work.  Our motivation is to have a quick check that can be
used during crawling. A high confidence is used to avoid false positives. 

## Design

* REST API based on python Flask
* Deep Learning neural networks 
  * Run with tensorflow serving for high throughput
  * CNN for image classification
  * BERT for text classification using a multilingual model
* FastText linear classifier
  * Full text 'bag of words' classification for high-throughput
  * URL-based classification
* PDF training data [preparation scripts](data_prep/README.md) for each kind of sub-classifier

Two other repos are relied upon and not copied into this repo because they are
useful standalone. This repo can be considered the 'roll up' that integrates
the ensemble.  

This PDF classifier can be re-purposed for other binary cases simply by using
different training data.  

### PDF Challenges

PDFs have challenges, because they can: 
* be pure images of text with no easily extractable text at all 
* range from 1 page position papers to 100+ page theses
* have citations either at the end of the document, or interspersed as footnotes 

We decided to avoid using OCR for text extraction for speed reasons and because
of the challenge of multiple languages.  

We address these challenges with an ensemble of classifiers that use confidence
values to cover all the cases. There are still some edge cases, but incidence
rate is at most a few percent.

## API Configuration and Deployment

The following env vars must be defined to run this API service:

- `FT_MODEL` full path to the FastText model for linear classifier
- `FT_URL_MODEL` path to FastText model for URL classifier
- `TF_IMAGE_SERVER_URL` base API URL for image tensorflow-serving process
- `TF_BERT_SERVER_URL` base API URL for BERT tensorflow-serving process

### Backend Service Dependency Setup

These directions assume you are running in an Ubuntu Xenial (16.04 LTS) virtual
machine.

    sudo apt-get install -y poppler-utils imagemagick libmagickcore-6.q16-2-extra ghostscript netpbm gsfonts-other
    conda create --name pdf_trio python=3.7 --file requirements.txt
    conda activate pdf_trio

Edit `/etc/ImageMagick/policy.xml` to change: 

    <policy domain="coder" rights="none" pattern="PDF" />

To: 

    <policy domain="coder" rights="read" pattern="PDF" />

We expect imagemagick 6.x; when 7.x is used, the binary will not be called
convert anymore.

### Back-Backend Components for Serving

- fastText-python is used for hosting fastText models
- tensorflow-serving for image and BERT inference

#### Tensorflow-Serving via Docker

Tensorflow serving is used to host the NN models and we prefer to use the REST because that significantly 
reduces the complexity on the client side.  

- install on ubuntu (docker.io distinguishes Docker from some preexisting
  package 'docker', a window manager extension):
  - `sudo apt-get install docker.io`
- get a docker image from the docker hub  (might need to be careful about v1.x to v2.x transition)
  - `sudo docker pull tensorflow/serving:latest`
  - NOTE: I am running docker at system level (not user), so sudo is needed for operations, YMMV
- see what is running:
  - `sudo docker ps`
- stop a running docker, using the id shown in the docker ps output:
  - `sudo docker stop 955d531040b2`
- to start tensorflow-serving:
  - `./start_bert_serving.sh`
  - `./start_image_classifier_serving.sh`

## Training and Models

These are covered in detail under [data_prep](data_prep/README.md):
- fastText (needed for training)
- bert variant repo for training
- tf_image_classifier repo

### Models

Sample models for research-pub classification are available at Internet Archive
under <https://archive.org/download/pdf_trio_models>.

A handy python
[package](https://archive.org/services/docs/api/internetarchive/quickstart.html#downloading)
will fetch the files and directory structure (necessary for
tensorflow-serving).  You can use curl and carefully recreate the directory
structure, of course. The full set of models is 1.6GB.

Here is a summary of the model files, directories, and env vars to specify the paths:

| env var | size | path | Used By |
| ------- | ---- | ------- | ------- |
| BERT_MODEL_PATH | 1.3GB | pdf_trio_models/bert_models | start_bert_serving.sh |
| IMAGE_MODEL_PATH | 87MB | pdf_trio_models/pdf_image_classifier_model | start_image_classifier_serving.sh |
| FT_MODEL          | 600MB | pdf_trio_models/fastText_models/dataset20000_20190818.bin | start_api_service.sh |
| FT_URL_MODEL      | 202MB | pdf_trio_models/fastText_models/url_dataset20000_20190817.bin | start_api_service.sh |


See [Data Prep](data_prep/README.md) for details on preparing training data and how to train each classifier.

