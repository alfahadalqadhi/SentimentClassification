# -*- coding: utf-8 -*-
"""
Created on Thu May  1 15:53:51 2025

@author: alfah
"""
import csv, pickle
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, Trainer, BertForSequenceClassification, BertTokenizer, BertConfig
from tokenizers import BertWordPieceTokenizer
from datasets import Dataset
from numpy import array, nonzero
from TokenizeDataset import load_tokenized_dataset
import os



DATASET_PATH='./datasets/TwitterSentiment_V2/twitter_training.csv'
VOCAB_PATH='./vocabs/TwitterSentiment_V2/vocab.txt'







"""
Data import
"""


tokenized_dataset, tokenizer=load_tokenized_dataset(DATASET_PATH,VOCAB_PATH)



"""
Initialize the model
"""

config=BertConfig(num_labels=3)

model= BertForSequenceClassification(config)




model.resize_token_embeddings(len(tokenizer))



       
"""
Split data
"""

SPLIT_PATH='/'.join(DATASET_PATH.split('/')[:-1])+'/Split'

train_test = tokenized_dataset.train_test_split(test_size=0.2)
train_valid = train_test['train'].train_test_split(test_size=0.2)
train_dataset = train_valid['train']
valid_dataset = train_valid['test']
test_dataset= train_test['test']

if not os.path.exists(SPLIT_PATH): 
     os.makedirs(SPLIT_PATH)

with open(SPLIT_PATH+'/tweet_sentiment_training.pkl', 'wb') as file:
    pickle.dump(train_dataset, file)

with open(SPLIT_PATH+'/tweet_sentiment_verification.pkl', 'wb') as file:
    pickle.dump(valid_dataset, file)

with open(SPLIT_PATH+'/tweet_sentiment_test.pkl', 'wb') as file:
    pickle.dump(test_dataset, file)





"""
Load training data
"""
with open("./datasets/TwitterSentiment_V1/Split/tweet_sentiment_training.pkl", "rb") as file: 
   training=pickle.load(file)
    
with open("./datasets/TwitterSentiment_V1/Split/tweet_sentiment_verification.pkl", "rb") as file:
   verification=pickle.load(file)

with open("./datasets/TwitterSentiment_V1/Split/tweet_sentiment_test.pkl", "rb") as file:
   test=pickle.load(file)


train_dataloader = DataLoader(training, shuffle=True, batch_size=8)
valid_dataloader = DataLoader(verification, batch_size=8)
"""
Transformers implementation
"""



training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy ="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training,
    eval_dataset=verification,
)

trainer.train()



print(trainer.predict(test)[2]["test_loss"])














