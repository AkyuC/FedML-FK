import argparse
import logging
import os
from statistics import mean, median
import sys
import matplotlib.pyplot as plt
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from DIoT.Model.GRU import GRUNet
from DataPreprocessing.DataLoader import data_simple, load_data, load_data_labels
from sklearn.cluster import KMeans
from DIoT.DataPreprocessing.FeatureCluster import FeatureCluster

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='lr', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--client_id', type=int, default=1, metavar='NN',
                        help='id of server')
    
    parser.add_argument('--server_ip', type=str, default="192.168.10.188",
                        help='IP address of the FedAvg server')
    
    parser.add_argument('--server_port', type=int, default=1883,
                        help='MQTT port of the FedAvg server')

    parser.add_argument('--client_num_in_total', type=int, default=1, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=1, metavar='NN',
                        help='number of workers')

    parser.add_argument('--data_len', type=int, default=5000, metavar='N',
                        help='the length of data using to trian (default: 5000)')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=1, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=100,
                        help='how many round of communications we shoud use')

    args = parser.parse_args()
    return args


def load_model():
    model = GRUNet()
    model.load_state_dict(torch.load('model.ckpt', map_location=lambda storage, loc: storage))
    return model

def test_predict_window(model, test_data, test_data_labels, threshold=0.01, gamma=0.5, omiga=250):
    model.eval()
    true_negative = 0
    false_positive = 0
    true_positive = 0
    false_negative = 0
    s, s_predict = test_data
    p_s_list = []

    for idx, inp in enumerate(s):
        p = model(torch.Tensor([[inp]]))
        p_s = p[0][s_predict[idx]]
        if p_s > threshold:
            p_s_list.append(1)
        else:
            p_s_list.append(0)
    for idx in range(len(test_data) - omiga):
        pass

def test_predict_packet(model, test_data, test_data_labels, threshold=0.01):
    model.eval()
    true_negative = 0
    false_positive = 0
    true_positive = 0
    false_negative = 0
    out = torch.nn.Softmax(dim=1)
    true_begin_num = 0
    true_attack_num = 0

    s, s_predict = test_data
    model = model.to(device)

    benign_predict_list = [0]
    attack_predict_list = [0]
    true_result = []
    predict_result = []

    for idx, inp in enumerate(s):
        inp = torch.Tensor([inp]).to(device)
        p = model(inp)
        p = out(p)
        for idx_context, context in enumerate(s_predict[idx]):
            p_s = p[idx_context, context]
            if test_data_labels[idx*len(s_predict[idx])+idx_context] == 1:
                attack_predict_list.append(float(p_s))
                true_result.append(1)
                true_attack_num += 1
                if p_s < threshold:
                    predict_result.append(1)
                    true_positive += 1
                else:
                    predict_result.append(0)
                    false_negative += 1
            else:
                true_result.append(0)
                true_begin_num += 1
                benign_predict_list.append(float(p_s))
                if p_s < threshold:
                    predict_result.append(1)
                    false_positive += 1
                else:
                    predict_result.append(0)
                    true_negative += 1

    print("true_attack_num: " + str(true_attack_num) + ", true_begin_num: " + str(true_begin_num))
    print("mean benign_predict_list: " + str(mean(benign_predict_list)) + ", median benign_predict_list:" + str(median(benign_predict_list)))
    print(true_positive, false_positive, false_negative, true_negative)

    benign_predict_list = np.array(benign_predict_list)
    pd.DataFrame(benign_predict_list).to_csv("benign_predict_probility.csv", index=False)

    # # print(benign_predict_list)
    # plt.figure(figsize=(9, 7))
    # plt.plot(benign_predict_list)
    # # plt.show()
    # plt.savefig("./benign.png", dpi=300)
    # plt.cla()
    # plt.figure(figsize=(9, 7))
    # # print(attack_predict_list[-100:])
    # plt.plot(attack_predict_list)
    # # plt.show()
    # plt.savefig("./attack.png", dpi=300)
    # plt.cla()
    # plt.figure(figsize=(9, 7))
    # # print(attack_predict_list[-100:])
    # plt.plot(true_result,color='b')
    # plt.plot(predict_result,color='r')
    # # plt.show()
    # plt.savefig("./true_and_predict.png", dpi=300)

    # accuracy = ((true_positive) + (true_negative)) \
    #             / ((true_positive) + (true_negative) + (false_positive) + (false_negative))
    # precision = (true_positive) / ((true_positive) + (false_positive))
    # false_positive_rate = (false_positive) / ((false_positive) + (true_negative))
    # tpr = (true_positive) / ((true_positive) + (false_negative))
    # tnr = (true_negative) / ((true_negative) + (false_positive))

    # print(accuracy, false_positive_rate, tpr, tnr)

    # return accuracy, precision, false_positive_rate

if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    np.random.seed(0)
    torch.manual_seed(10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    id = "3.11"
    ip = "192.168." + id
    train_data = load_data("../Data/SYN_DoS_pcap%s.csv" % id)

    k = 20
    n_clusters = 100
    len_clusters = 10000
    test_data_num = 10000
    # 取最后 10000 个数据包进行测试
    # kM:KMeans = FeatureCluster(n_clusters, train_data[0:len_clusters])
    f = open("./kM",'rb')
    kM:KMeans = pickle.load(f)
    f.close()
    test_data_all = kM.fit(train_data)
    # data_simple(kM.fit(train_data[len_clusters:len_clusters + train_data_num]).labels_.tolist(), k, args.batch_size)
    # test_data = data_simple(test_data_all.labels_.tolist()[810375-5000:810375+5000], k, args.batch_size)
    # test_data_labels_all = load_data_labels("../Data/SYN_DoS_labels%s.csv" % id)
    # test_data_labels = test_data_labels_all[810375-5000+k:810375+5000+k]

    test_data = data_simple(test_data_all.labels_.tolist()[20000:20000+10000], k, args.batch_size)
    test_data_labels_all = load_data_labels("../Data/SYN_DoS_labels%s.csv" % id)
    test_data_labels = test_data_labels_all[20000+k:20000+10000+k]
    
    # print(sum(test_data_labels)*64)
    # print(len(test_data))

    model = load_model()

    print("packet detection:")
    threshold = 0.01
    test_predict_packet(model, test_data, test_data_labels, threshold)
    
    # print("packet window detection:")
    # test_data = data_simple(test_data_all.labels_.tolist(), k, n_clusters)[len_clusters:len_clusters+test_data_num]
    # test_data_labels = test_data_labels_all[len_clusters+k:len_clusters+test_data_num+k]
    # test_predict_window(model, test_data, test_data_labels)
