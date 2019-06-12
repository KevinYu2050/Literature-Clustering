import os
import re


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

    booklist =[list(filter(None, [sent.replace('\n', '') for sent in book])) for book in booklist]
    booklist =[[sent.replace('\xa0', '') for sent in book] for book in booklist]

    for book in booklist:
        for sent in book:
            if re.match(start_pattern, sent):
                index = book.index(sent)
                del book[:index+1]
            if re.match(end_pattern, sent):
                index = book.index(sent)
                del book[index:]
            if sent == '':
                del sent
    

    return booklist 


print(preprocessing(read_files('./gutenberg_shakespeare/')))