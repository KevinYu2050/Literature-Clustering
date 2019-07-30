import pandas as pd
import collections
def modify(data_dir, label_dir):
    with open(data_dir, encoding='utf8') as f:
        texts = f.readlines()

    with open(label_dir, encoding='utf8') as f:
        labels = f.readlines()
    
    # labels_rep = [label for label in labels]
    data_rep = []
    labels_rep = []
    counter = counter0 = counter1 = counter2 = counter3 = counter4 = counter5 = counter6 = counter7 = \
    counter8 = counter9 = counter10 =counter11 = counter12 = counter13 = counter14 = counter15= \
        counter16 = counter17= 0
    
    for label in labels:
        if label == 'Adventure\n':
            if counter0 <= 1500:
                counter0 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Satire\n':
            if counter1 <= 1500:
                counter1 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Reference\n':
            if counter2 <= 1500:
                counter2 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Historical Fiction\n':
            if counter3 <= 1500:
                counter3 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass    
        elif label == 'Fantasy\n':
            if counter4 <= 1500:
                counter4 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass        
        elif label == 'Romance\n':
            if counter5 <= 1500:
                counter5 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Memoirs\n':
            if counter6 <= 1500:
                counter6 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Classics\n':
            if counter7 <= 1500:
                counter7 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Humor\n':
            if counter8 <= 1500:
                counter8 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Crime & Mystery\n':
            if counter9 <= 1500:
                counter9 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Science Fiction\n':
            if counter10 <= 1500:
                counter10 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Biography\n':
            if counter11 <= 1500:
                counter11 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Tragedy\n':
            if counter12 <= 1500:
                counter12 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Fictional Diaries\n':
            if counter13 <= 1500:
                counter13 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Philosophy\n':
            if counter14 <= 1500:
                counter14 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Horror\n':
            if counter15 <= 1500:
                counter15 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Analysis\n':
            if counter16 <= 1500:
                counter16 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass
        elif label == 'Economics\n':
            if counter17 <= 1500:
                counter17 += 1
                labels_rep.append(label)
                data_rep.append(texts[counter])
            else:
                pass  
        counter += 1
                                                               
    # data = {'genre':labels, 'text':texts}
    # dataframe = pd.DataFrame(data, index = [i for i in range(len(data['genre']))], columns = ['genre', 'text'])
    # print(dataframe)

    return data_rep, labels_rep

def save_file(data, labels, data_dir, label_dir):
    with open(data_dir, 'w+', encoding="utf-8") as f:
        for i in data:
            f.write(i)

    with open(label_dir, 'w+', encoding="utf-8") as f:
        for i in labels:
            f.write(i)

data, labels = modify(r'C:\Users\kk\Desktop\Pioneer\dataset_for_multiclass_classification.txt',
        r'C:\Users\kk\Desktop\Pioneer\labels_for_multiclass_classification.txt')
print(len(data), len(labels))
save_file(data, labels, r'C:\Users\kk\Desktop\Pioneer\dataset_for_multiclass_classification_test_modified.txt',
    r'C:\Users\kk\Desktop\Pioneer\labels_for_multiclass_classification_test_modified.txt')
with open(r'C:\Users\kk\Desktop\Pioneer\labels_for_multiclass_classification_test_modified.txt', encoding='utf8') as f:
        labels = f.readlines()
print(collections.Counter(labels))