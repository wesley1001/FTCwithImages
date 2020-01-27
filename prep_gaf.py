from pyts.image import GramianAngularField
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from labeling import triple_barrier_labeling
from params import INPUT_PARAMS
gaf = GramianAngularField()


def transform(series, transformer):
    array = transformer.transform(series)
    return array[0]


def transform_gaf(data_list):
    transformed_data_list = []
    for tup in tqdm(data_list):
        tr_ar = transform(np.reshape(tup[0].values, (1, -1)), gaf)
        label = tup[1]
        transformed_data_list.append((tr_ar, label))
    return transformed_data_list


def make_save_plots(series, label, path, name):
    img = plt.imshow(series)
    img.set_cmap('rainbow')
    plt.axis('off')
    plt.savefig(os.path.join(path, f"{name}.png"),
                bbox_inches='tight',
                pad_inches=0)
    plt.clf()


def save_train_val(transformed_data_list, general_data_path):
    os.makedirs(general_data_path, exist_ok=True)
    train_path = os.path.join(general_data_path, 'train')
    os.makedirs(train_path,
                exist_ok=True)
    val_path = os.path.join(general_data_path, 'val')

    for label in ['-1', '0', '1']:
        os.makedirs(os.path.join(val_path, label), exist_ok=True)
        os.makedirs(os.path.join(train_path, label), exist_ok=True)
    os.makedirs(val_path,
                exist_ok=True)
    train, val = train_test_split(transformed_data_list)
    i = 0

    for tup in tqdm(train):
        label = tup[1]
        path = os.path.join(train_path, str(label))
        make_save_plots(tup[0], label, path, str(i))
        i += 1

    for tup in tqdm(val):
        label = tup[1]
        path = os.path.join(val_path, str(label))
        make_save_plots(tup[0], label, path, str(i))
        i += 1


def run(source_path, destination_dir):
    data = pd.read_csv(source_path)
    data = data.fillna(method='bfill')
    data_list = triple_barrier_labeling(data.Value, data.Value, data.Value,
                                        data.Value, INPUT_PARAMS)
    print(pd.DataFrame([f[1] for f in data_list])[0].value_counts())
    transformed_data_list = transform_gaf(data_list)
    save_train_val(transformed_data_list, destination_dir)


