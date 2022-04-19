import os
import sys

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))


def load_data(file:str, batch_size:int):
    train_data_iter = torch.utils.data.DataLoader(np.array(pd.read_csv(file)), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_data_iter


if __name__ == "__main__":
    id = "3.11"
    ip = "192.168." + id
    data = load_data("../Data/SYN DoS_pcap%s.csv" % id)
    print("finished!")