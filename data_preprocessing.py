import os
import re
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
        with open(dir + f, 'r') as f:
            content.append(f.readlines())
    
    return content

def preprocessing(booklist):
    # Remove all the licenses and text-unrelated content
    start_pattern = '[***] START OF THIS PROJECT GUTENBERG EBOOK'
    end_pattern = 'End of Project Gutenberg'

    booklist = [list(filter(None, [sent.replace('\n', '') for sent in book])) for book in booklist]
    booklist = [[sent.replace('\xa0', '') for sent in book] for book in booklist]
    booklist = [[sent for sent in book if not re.match('^[^a-z]*$', sent)] for book in booklist]

    for book in booklist:
        for sent in book:
            if re.match(start_pattern, sent):
                index = book.index(sent)
                del book[:index+1]
            elif re.match(end_pattern, sent):
                index = book.index(sent)
                del book[index:]
    

    return booklist 


def create_dataset(booklist, label):
    # Divide the dataset into basic units of (sent_data, label) tuples
    new_list = []
    for book in booklist:
        from_ = int(len(book)/10)
        to = int(9 * len(book)/10)
        for sent in book[from_:to]:
            new_list.append((sent, label))

    return new_list


# Save the processed data into new txt files to fit in the tensorflow.data pipeline
skp = create_dataset(preprocessing(read_files('./gutenberg_shakespeare/')), 'Shakespearean')
non_skp = create_dataset(preprocessing(read_files('./gutenberg/')), 'Non-Shakespearean')

dataset = skp + non_skp[:len(skp)]
random.shuffle(dataset)
with open('./data.txt', 'w+', encoding="utf-8") as f:
    for i in dataset:
        f.write(i[0])
        f.write('\n')

with open('./label.txt', 'w+', encoding="utf-8") as f:
    for i in dataset:
        f.write(i[1])
        f.write('\n')
