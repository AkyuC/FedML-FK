import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

def load_data(file:str):
    return np.array(pd.read_csv(file))

def load_data_labels(file:str):
    return np.array(pd.read_csv(file))


if __name__ == "__main__":
    id = "3.11"
    ip = "192.168." + id
    data = load_data("../Data/SYN DoS_pcap%s.csv" % id)
    print("finished!")