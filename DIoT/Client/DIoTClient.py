import argparse
import os
import sys
import time

import numpy as np
import torch.nn
from DIoT.DataPreprocessing.FeatureCluster import FeatureCluster
from DIoT.ClientManager.DIoTClientManager import DIoTClientManager

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

# from FedAvg.FedAvgClient.FedAvgClientManager import FedAvgClientManager
from Model.GRU import GRUNet
from Trainer.GRUTrainer import GRUTrainer

from DataPreprocessing.DataLoader import load_data


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


if __name__ == '__main__':
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    np.random.seed(0)
    torch.manual_seed(10)

    device = torch.device("cpu")

    # load data
    id = "3.11"
    ip = "192.168." + id
    train_data = load_data("../Data/SYN DoS_pcap%s.csv" % id)
    train_data_num = len(train_data)

    k = 20
    n_clusters = 100
    kM = FeatureCluster(n_clusters, train_data)
    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model =  GRUNet()

    # start training    
    trainer = GRUTrainer(model, args)

    client_manager = DIoTClientManager(args, args.client_id, trainer, train_data, train_data_num, k, kM, n_clusters, device)
    client_manager.run()
    client_manager.start_training()

    while(client_manager.is_finish is False):
        time.sleep(5)
