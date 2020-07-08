# An Efficient Framework for Vietnamese Sentiment Analysis
This is a Pytorch implementation for the paper "An Efficient Framework for Vietnamese Sentiment Analysis", which is accepted at the conference SOMET 2020.

## Requirement
* python                    3.7.3
* pytorch                   1.4.0
* pytorch-transformers      1.2.0
* tensorflow                2.0.0
* torchtext                 0.4.0
* torchvision               0.4.0
* scikit-image              0.15.0
* scikit-learn              0.20.3
* nltk                      3.4.5
* fairseq                   0.9.0
* vncorenlp                 1.0.3


## Data preparation
* In this work, we use two datasets:
  * AIVIVN: this is the publish dataset from AIVIVN 2019 Sentiment Challenge, including approximately 160K training reviews with the available labels and 11K testing reviews without the available labels. We manually did labelling for the testing dataset.
  * Our dataset: this is our new dataset which was crawled from the Vietnamese e-commerce websites, the reviews are started from Jan 2019 and includes all product categories. We trained all the methods with 10K, 15K, 20K training reviews respectively and tested on about 170K reviews.
* The validation dataset is randomly selected from the training dataset, with 20%.
* The two datasets are placed at the folders */dataset/aivivn/ and */dataset/tiki/.

## Pre-trained Models preparation
* BERT: which is the pre-trained BERT model with the version of bert-base-multilingual-uncased and automatically downloaded from Huggingface Transformers.
* PhoBERT: which is the state-of-the-art pre-trained BERT model for the Vietnamese language. To run with the pre-trained PhoBert models, we need to do:
  * Download https://public.vinai.io/PhoBERT_base_transformers.tar.gz, extract and place at */phobert
  * Download vncorenlp from https://github.com/VinAIResearch/PhoBERT#vncorenlp, extract and place at */vncorenlp

## Training
* Do configuration for aivivn and tiki datasets at */config/aivivn/ and */config/tiki/ respectively.
* Run run.sh for training all methods and datasets.

## Citation
Please cite our paper if VietnameseSentimentFramework is used:
```
@article{VietnameseSentimentFramework,
  title={An Efficient Framework for Vietnamese Sentiment Analysis},
  author={Cuong V. Nguyen, Khiem H. Le, Anh M. Tran, Binh T. Nguyen},
  journal={Proceedings of The 18th International Conference on Intelligent Software Methodologies, Tools, and Techniques (SoMeT) 2020},
  year={2020}
}
```
If this implementation is useful, please cite or acknowledge this repository on your work.

## Contact
Cuong V. Nguyen (cuong.vn08@gmail.com),

Khiem H. Le (lehuykhiem28011999@gmail.com),

Anh M. Tran (trminhanh115@gmail.com),

Binh T.Nguyen (ngtbinh@hcmus.edu.vn)
