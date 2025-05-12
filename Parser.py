# -*- coding: utf-8 -*-
"""
Created on Thu May  8 12:46:12 2025

@author: alfah
"""

import argparse

def main_parse_args(*args):
    
    """
         A function for parsing the arguments passed to the file. 
         TODO: add options to configure the model.
    """
    
    parser= argparse.ArgumentParser(description= "An implementation of BERT text classification.")
    
    #Add optional arguments
    
    parser.add_argument('--dataset',
                        type=str,
                        help= 'A string for the path to the dataset.csv file.\
                               The dataset should be in csv format with one\
                               field containing the text and the second containing\
                               its class. The text must be a string and the\
                               class could be a string (ex. sentiment) or int.' ,
                        default= './datasets/TwitterSentiment_V1/tweet_sentiment.csv')
    parser.add_argument('--vocab',
                        type=str,
                        help= '"from_pretrained" or a path to the vocab.txt\
                                file for the tokenizer containing one word per\
                                line. You can use a vocabulary of your own,\
                                if the file does not exist at the path a\
                                vocabulary will be created based on the words\
                                in the dataset. Default: "from_pretrained"',
                        default= 'from_pretrained')
    
    
    
    
    return parser.parse_args(*args)
    
    