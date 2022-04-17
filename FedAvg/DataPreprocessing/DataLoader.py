import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))


def load_data(simple_len = 5000, batch_size = 64):

    benign_data = pd.read_csv('../Data/benign_traffic.csv')

    try:
        min = np.loadtxt('../Data/min.txt')
        max = np.loadtxt('../Data/max.txt')
    except IOError:
        np.savetxt('../Data/min.txt', np.array(benign_data).min(axis=0))
        np.savetxt('../Data/max.txt', np.array(benign_data).max(axis=0))
        min = np.loadtxt('../Data/min.txt')
        max = np.loadtxt('../Data/max.txt')

    benign_data = benign_data[:simple_len]
    benign_data = np.array(benign_data)
    benign_data[np.isnan(benign_data)] = 0
    benign_data = (benign_data - min) / (max - min)
    
    train_data_iter = torch.utils.data.DataLoader(benign_data, batch_size=batch_size, shuffle=False, num_workers=0)
    train_data_num = round(len(train_data_iter) * 2 / 3) * batch_size

    return train_data_num, train_data_iter


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
    # logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    dataset = load_data(5000, 64)
    [train_data_num, train_data_iter] = dataset
    print("end!")