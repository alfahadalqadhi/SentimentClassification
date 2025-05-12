# -*- coding: utf-8 -*-
"""
Created on Thu May  1 15:53:51 2025

@author: alfah
"""
import pickle, os
from transformers import TrainingArguments, Trainer, BertForSequenceClassification, BertConfig
from TokenizeDataset import load_tokenized_dataset
from Parser import main_parse_args
from CustomCallback import CustomCallback, compute_metrics

# Parse passed arguments, refer to Parser for more information

args=main_parse_args()


if args.dataset and args.vocab:
    DATASET_PATH=args.dataset
    VOCAB_PATH=args.vocab








#Import data, tokenize the text, and replace the classes with integers

tokenized_dataset, tokenizer=load_tokenized_dataset(DATASET_PATH,VOCAB_PATH)


#Initialize the model

config=BertConfig(num_labels=max(tokenized_dataset['label'])+1)

model= BertForSequenceClassification(config)

model.resize_token_embeddings(len(tokenizer))


#Split data


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


#Load training data

with open(SPLIT_PATH+'/tweet_sentiment_training.pkl', "rb") as file: 
   training=pickle.load(file)
    
with open(SPLIT_PATH+'/tweet_sentiment_verification.pkl', "rb") as file:
   verification=pickle.load(file)

with open(SPLIT_PATH+'/tweet_sentiment_test.pkl', "rb") as file:
   test=pickle.load(file)





#Transformers implementation

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.1,
    save_strategy ="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training,
    eval_dataset=verification,
    compute_metrics=compute_metrics,
    processing_class=tokenizer
)

trainer.add_callback(CustomCallback(trainer))

trainer.train()

trainer.save_model("./results/"+SPLIT_PATH.split('/')[2])

#Print out of sample results

print(trainer.predict(test)[2])