import rnn_reader_keras as reader
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd 
import os
import collections
import pickle
import tensorflow as tf 
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_mutual_info_score

# def get_batch(data, batch_size):
#     # Read the data by batches 
#     n_batches = len(data)/batch_size 
#     remainder = len(data)%batch_size
#     # subarrays = np.split(data[:(len(data)-remainder)], n_batches)
#     subarrays = np.split(data, n_batches)

#     return subarrays

def create_data(data_dir, label_dir, model_dir, to_dir):
    # Encode the data using the trained autoencoder
    model = tf.keras.models.load_model(model_dir)

    x_train, y_train, _, _, _, _, encoder = reader.read_data(data_dir, label_dir, 20000)

    get_encoded = tf.keras.backend.function([model.layers[0].input],
     [model.layers[3].output])
    x_encoded = get_encoded(x_train)
    x_encoded = x_encoded[0].reshape(x_encoded[0].shape[0], x_encoded[0].shape[1] * x_encoded[0].shape[2])
    np.save(to_dir, x_encoded)

    return y_train, encoder

def cluster(data_dir, y_train, n_clusters):
    # Uses K-Means to cluster the encoded vectors
    
    # Read the previously encoded numpy arrays
    for f in os.listdir(data_dir):
        if f.endswith('npy'):
            try:
                x_train = np.load(data_dir + f)
            except MemoryError:
                pass
                break
        print('loading ' + f)

    print(len(x_train), len(y_train))
    data = {'genre':y_train, 'text':x_train}

    km = KMeans(init='k-means++', n_clusters=n_clusters)
    print('training the model')
    clf = km.fit(x_train)

    clusters = clf.labels_.tolist()
    dataframe = pd.DataFrame(data, index = [clusters], columns = ['genre'])

    pickle.dump(clf, open('km_clustering_modified{}.sav'.format(n_clusters), 'wb'))
    dataframe.to_pickle('df_clustering_modified{}.pkl'.format(n_clusters))


    return 

def cluster_plot(model_dir, to_dir, n_clusters):
    # Directly plot the clustering results of the model
    km = pickle.load(open(model_dir, 'rb'))

    fig = plt.figure(figsize=(15,6))
    plt.hist(km.labels_, bins=n_clusters)
    plt.xticks(range(n_clusters))
    # plt.show()
    fig.savefig(to_dir, dpi = fig.dpi)

    return

def cluster_count(data_dir, encoder, to_dir, n_clusters):
    data = pd.read_pickle(data_dir)
    all_cluster_count = []

    for i in range(n_clusters):
        c = collections.Counter(encoder.inverse_transform(data.ix[i]['genre'].sort_values(ascending=True).values.tolist()))
        all_cluster_count.append(c.most_common(5))

    columns = ['1st', '2nd', '3rd', '4th', '5th']
    df = pd.DataFrame(all_cluster_count, index = ['Cluster' + str(i) for i in range(n_clusters)], columns=columns)
    df.to_csv(to_dir)

    return

def scores(data_dir, model_dir, labels_true):
    # Using several differeent scales of scores to evaluate the results
    X = np.load(data_dir)
    km = pickle.load(open(model_dir, 'rb'))
    labels_predicted = km.labels_
    s_score = silhouette_score(X, labels_predicted, metric='euclidean')
    
    # y_train = labels_true.tolist()
    nmi = normalized_mutual_info_score(labels_true, labels_predicted)
    print(y_train)
    print(labels_predicted)

    return s_score, nmi

def plot_tsne():
    



# y_train, encoder = create_data(r'C:\Users\kk\Desktop\Pioneer\dataset_for_multiclass_classification_test_modified.txt',
#         r'C:\Users\kk\Desktop\Pioneer\labels_for_multiclass_classification_test_modified.txt', 
#         r'C:\Users\kk\Desktop\Pioneer\keras_autoencoder.h5',
#         r'C:\Users\kk\Desktop\Pioneer\Encoded_data_modified\x_encoded')

_, y_train, _, _, _, _, encoder = reader.read_data('dataset_for_multiclass_classification_test_modified.txt',
        'labels_for_multiclass_classification_test_modified.txt', 20000)
all_s_score = []
all_nmi = []
for i in range(20, 25):
    print('training model {}'.format(i))
    cluster('Encoded_data_modified\\', y_train, i)

    cluster_plot('km_clustering_modified{}.sav'.format(i),
        'cluster_plot_modified{}.png'.format(i), i)

    cluster_count('df_clustering_modified{}.pkl'.format(i), encoder, 'all_collections_count_df_modified{}.csv'.format(i), i)

for i in range(10, 25):
    s, nmi = scores('Encoded_data_modified\\x_encoded.npy','km_clustering_modified{}.sav'.format(i), y_train)
    all_s_score.append((i, s))
    all_nmi.append((i, nmi))

# fig = plt.figure(figsize=(15,6))
# plt.plot(*zip(*all_s_score), color='red', label='silhouette_score')
# plt.plot(*zip(*all_nmi), color='blue', label = 'nmi score')
# plt.title('Evaluation for models initiated with different numbers of clusters')
# plt.xlabel('number of clusters')
# plt.legend()
# fig.savefig('different clusters.png', dpi = fig.dpi)

fig = plt.figure(figsize=(15,6))
plt.plot(*zip(*all_s_score), color='red')
plt.title('Silhouette Scores for models with different numbers of clusters')
plt.xlabel('number of clusters')
fig.savefig('Silhouette Scores.png', dpi = fig.dpi)

fig = plt.figure(figsize=(15,6))
plt.plot(*zip(*all_nmi), color='red')
plt.title('NMI scores for models with different numbers of clusters')
plt.xlabel('number of clusters')
fig.savefig('Nmi Scores.png', dpi = fig.dpi)