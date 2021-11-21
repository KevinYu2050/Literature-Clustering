import os
import re
import string
import random
import pandas as pd 
import numpy as np

def read_files(dir):
    content = []

    # Read the local file names into and save them into a list
    books = [f for f in os.listdir(dir) 
        if f.endswith('.txt')]
    # Read the files into a list    
    for f in books:
        with open(dir + f, 'r', encoding='utf-8') as file:
            content.append((f.split('_')[0], file.readlines()))
    
    return content

def preprocessing(book):
    # Remove all the licenses and text-unrelated content
    start_pattern = '[***] START OF THIS PROJECT GUTENBERG EBOOK'
    end_pattern = 'End of Project Gutenberg'
    # punc_pattern = string.punctuation + '\n' 

    book = [re.sub(' +', ' ', sent) for sent in book] 
    book = [re.sub('\d+', '', sent) for sent in book]   
    book = [sent for sent in book if not re.match('^[^a-z]*$', sent)]
    book = [sent.replace('\xa0', '') for sent in book] 
    book = [sent.lower() for sent in book] 
    book = list(filter(None, [sent.replace('\n', '') for sent in book])) 


    # book = [re.sub('[{}]'.format(punc_pattern), r'', sent).lower() for sent in book] 

    for sent in book:
        if re.match(start_pattern, sent):
            index = book.index(sent)
            del book[:index+1]
        elif re.match(end_pattern, sent):
            index = book.index(sent)
            del book[index:]
    

    return book


def create_dataset(booklist):
    # Divide the dataset into basic units of (sent_data, label) tuples
    new_list = []
    for book in booklist:
        from_ = int(len(book[1])/10)
        to = from_ + 10000
        for sent in book[1][from_:to]:
            new_list.append((sent, book[0]))

    return new_list


# Save the processed data into new txt files to fit in the tensorflow.data pipeline
booklist = [(book[0], preprocessing(book[1])) for book in read_files('../gutenberg_preprocessed/')]

dataset = create_dataset(booklist)
random.shuffle(dataset)

with open('../processed_datasets/dataset_for_multiclass_classification_test_modified.txt', 'w+', encoding="utf-8") as f:
    for i in dataset:
        f.write(i[0])
        f.write('\n')

with open('../processed_datasets/labels_for_multiclass_classification_test_modified.txt', 'w+', encoding="utf-8") as f:
    for i in dataset:
        f.write(i[1])
        f.write('\n')

# booklist, _ = read_files('./gutenberg/')
# booklist_new = [(book[0], preprocessing(book[1])) for book in booklist]
# counter = 1
# for book in booklist_new:
#     from_ = int(len(book[1])/10)
#     to = int(9 * len(book[1])/10)
#     with open('./gutenberg_preprocessed/{}:{}.txt'.format(book[0], counter), 'w+', encoding="utf-8") as f:
#         for line in book[1][from_:to]:
#             f.write(line)
#             f.write('\n')
#     counter += 1


