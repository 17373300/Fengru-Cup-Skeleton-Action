import numpy as np
import pickle
import os

dataset_types = ['xsub', 'xview']

for type in dataset_types:
    folderName = os.path.join('data', 'NTU-RGB+D', type)
    train_data_fileName = os.path.join(folderName, 'train_data.npy')
    train_label_fileName = os.path.join(folderName, 'train_label.pkl')
    val_data_fileName = os.path.join(folderName, 'val_data.npy')
    val_label_fileName = os.path.join(folderName, 'val_label.pkl')

    train_data = np.load(train_data_fileName)
    with open(train_label_fileName, 'rb') as f:
        train_label = pickle.load(f)




