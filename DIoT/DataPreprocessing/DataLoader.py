import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

def load_data(file:str):
    return np.array(pd.read_csv(file))

def load_data_labels(file:str):
    return np.array(pd.read_csv(file))

def data_simple(data:list, input_dim, output_dim):
    train_dataset_input = []    # input_dim data
    train_dataset_predict  = [] # next_predict data
    data_len = len(data)
    for index in range(data_len - input_dim):
        # get once input data
        once_input = data[index:index+input_dim]
        train_dataset_input.append(once_input)
        # get once predict data
        # predict_target = [0 for _ in range(output_dim)]
        predict_target = data[index+input_dim]
        train_dataset_predict.append(predict_target)
    return train_dataset_input, train_dataset_predict
    

if __name__ == "__main__":
    id = "3.11"
    ip = "192.168." + id
    data = load_data("../Data/SYN DoS_pcap%s.csv" % id)
    print("finished!")