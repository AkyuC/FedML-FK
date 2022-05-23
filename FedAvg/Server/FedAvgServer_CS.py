import logging
import os
import sys
import argparse
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from FedAvgServerManager.FedAvgAggregator_CS import FedAVGAggregator
from FedAvgServerManager.FedAvgServerManager_CS import FedAVGServerManager
from Model.AutoEncoder import AutoEncoder
from Trainer.AETrainer import AETrainer


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='lr', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--server_id', type=int, default=0, metavar='NN',
                        help='id of server')
    
    parser.add_argument('--server_ip', type=str, default="192.168.10.188",
                        help='IP address of the FedAvg server')
    
    parser.add_argument('--server_port', type=int, default=1883,
                        help='MQTT port of the FedAvg server')

    parser.add_argument('--client_num_in_total', type=int, default=7, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=7, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=6,
                        help='how many round of communications we shoud use')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    device_id_to_client_id_dict = dict()

    logging.info(args)

    # Set the random seed.
    np.random.seed(0)
    torch.manual_seed(10)

    device = torch.device("cpu")

    # create model
    model = AutoEncoder()
    model_trainer = AETrainer(model)

    aggregator = FedAVGAggregator(args, args.client_num_per_round, device, model_trainer)

    size = args.client_num_per_round + 1
    server_manager = FedAVGServerManager(args, aggregator, args.server_id, args.client_num_per_round, 
                                         args.server_ip, args.server_port, topic="fediot")
    server_manager.run()

    while(server_manager.is_finish is False):
        time.sleep(5)
