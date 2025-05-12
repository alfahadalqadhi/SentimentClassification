# -*- coding: utf-8 -*-
"""
Created on Wed May  7 18:20:40 2025

@author: alfah
"""


import os
import csv
from datasets import Dataset
from transformers import BertTokenizer
from tqdm import tqdm




def label2int(row, groups):
    
    """
         A helper function that assigns integers to classes.
         Parameters
         ----------
         row: dict {text: any, label: str or int}
             A row from the dataset.
         groups: set(str or int)
             The distict groups in the dataset.
    """
    
    
    for N in range(len(groups)):
        if row['label']==groups[N]:
            return {'label':N}



def collect_vocab(strings,VOCAB_PATH):
    
    """
         A helper function that creates a simple costum vocabulary for the encoder.
         Using this will improve the computational speed but may reduce the accuracy.
         Must provide a path for the vocabulary to be stored at.
         Parameters
         ----------
         strings: [str]
             A list containing all the text from the dataset.
         VOCAB_PATH: str
             The path to the file where the vocabulary is to be stored if not using "from_pretrained".
             Note that if the file already exists this function is skipped and the vocabulary will be loaded instead.
    """
    
    unique_words=[]
    with open(VOCAB_PATH,"w", newline="\n", encoding='utf-8') as vocab:
        for string in tqdm(strings):
            words=string.split()
            for word in words:
                if(word not in unique_words):
                    unique_words+=[word]
                    vocab.write(word+"\n")



def tokenize_function(examples, tokenizer):
    
    """
          A helper function that tokenizes text in batches.
          Parameters
          ----------
          examples: [dict({text: str,...})]
              A list containing all the text from the dataset.
          tokenizer: Tokenizor
              A Tokenizor object visit: https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/tokenizer#tokenizer for more information
     """   
    
    
    return tokenizer(examples['text'], padding='max_length', truncation=True)




    
def load_tokenized_dataset(DATASET_PATH,VOCAB_PATH):
    
    """
          The main function that loads and processes the dataset in preparation for the training.
          Parameters
          ----------
          DATASET_PATH: str
              Path to the .csv dataset file
          VOCAB_PATH: str
              Path to the .txt vocabulary file
    """
    
    
    dataset_dict= {"text":list(),
                   "label":list()}
    if not os.path.exists('/'.join(DATASET_PATH.split('/')[:-1])):
        os.makedirs('/'.join(DATASET_PATH.split('/')[:-1]))
    
    with open(DATASET_PATH, newline='\n', errors='ignore') as csvfile:
        dataset_reader = csv.DictReader(csvfile)
        dictkeys=dataset_reader.fieldnames
        for row in dataset_reader:
            text=dataset_dict["text"]+[row[dictkeys[0]]]
            label=dataset_dict["label"]+[row[dictkeys[1]]]
            dataset_dict={'text':text,
                          'label':label}
            
    dataset=Dataset.from_dict(dataset_dict)        
            
    groups=list(set(dataset['label']))
    
    dataset=dataset.map(lambda x: label2int(x,groups))
    if VOCAB_PATH=='from_pretrained':
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    else:
        if not os.path.exists('/'.join(VOCAB_PATH.split('/')[:-1])):
            os.makedirs('/'.join(VOCAB_PATH.split('/')[:-1]))
        
        if os.path.exists(VOCAB_PATH):
            tokenizer = BertTokenizer(vocab_file=VOCAB_PATH, unk_token='<unk>')
            tokenizer.model_max_length=1000
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
        else:
            collect_vocab(dataset['text'], VOCAB_PATH)
            tokenizer = BertTokenizer(vocab_file=VOCAB_PATH)
            tokenizer.model_max_length=1000
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
    
        
    if tokenizer:
        tokenized_dataset= dataset.map(lambda x: tokenize_function(x,tokenizer), batched=True)
    else:
        raise Exception('The tokenizor failed to initialize.')
    
    return [tokenized_dataset,tokenizer]