import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

def load_data(file:str):
    return np.array(pd.read_csv(file))

def load_data_labels(file:str):
    return np.array(pd.read_csv(file))

def data_simple(data:list, input_dim, batch_size):
    train_dataset_input = []    # input_dim data
    train_dataset_predict  = [] # next_predict data
    data_len = len(data)
    train_data_batch_list = []
    train_data_labels_batch_list = []
    for index in range(data_len - input_dim):
        random_pick = random.randint(0,data_len-input_dim -1)
        # get once input data
        once_input = data[random_pick:random_pick+input_dim]
        train_data_batch_list.append(once_input)
        # get once predict data
        # predict_target = [0 for _ in range(output_dim)]
        predict_target = data[random_pick+input_dim]
        train_data_labels_batch_list.append(predict_target)
        if (index + 1) % (batch_size) == 0:
            train_dataset_input.append(train_data_batch_list)
            train_dataset_predict.append(train_data_labels_batch_list)
            train_data_batch_list = []
            train_data_labels_batch_list = []
    tmp = list(zip(train_dataset_input, train_dataset_predict))
    random.shuffle(tmp)
    train_dataset_input, train_dataset_predict = zip(*tmp)
    return train_dataset_input, train_dataset_predict
    

if __name__ == "__main__":
    id = "3.11"
    ip = "192.168." + id
    data = load_data("../Data/SYN DoS_pcap%s.csv" % id)
    print("finished!")