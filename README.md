# SentimentClassification
A sample project for sentiment classification using the Transformers implementation of the BERT language model.
## Requirements
To run this project you will need to have [Docker](https://docs.docker.com/desktop/) and [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) installed.
## Installation and testing
1. Clone the repo
```
git clone https://github.com/alfahadalqadhi/SentimentClassification
```
2. Compose the docker container
```
docker compose up
``` 
3. Follow the link given after the image is built and running. This should lead you to the jupyter-lab page, where you can access the terminal of the image.
4. In the terminal you can run the classify.py file to check that everything is working properly.
```
python classify.py -h
```
This should print how to use classify.py on your own dataset. You can test the code on the default dataset
```
python classify.py
```
or by providing the path to one of the other .csv files
```
python classify.py --dataset <path>
```
Note however the larger datasets will take a longer time.
