import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from Model.AutoEncoder import AutoEncoder


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

def load_data(batch_size=64, client_id=1):
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    th_local_dict = dict()

    client_data = ['Danmini_Doorbell', 'Ecobee_Thermostat', 
        'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor',
        'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera', 
        'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
        'SimpleHome_XCS7_1003_WHT_Security_Camera']
        
    benign_data = pd.read_csv('../Data/UCI-MLR/'+client_data[client_id-1]+'/benign_traffic.csv')
    benign_data = np.array(benign_data)

    try:
        min = np.loadtxt('../Data/UCI-MLR/'+client_data[client_id-1]+'/min.txt')
        max = np.loadtxt('../Data/UCI-MLR/'+client_data[client_id-1]+'/max.txt')
    except IOError:
        np.savetxt('../Data/UCI-MLR/'+client_data[client_id-1]+'/min.txt', np.array(benign_data).min(axis=0))
        np.savetxt('../Data/UCI-MLR/'+client_data[client_id-1]+'/max.txt', np.array(benign_data).max(axis=0))
        min = np.loadtxt('../Data/UCI-MLR/'+client_data[client_id-1]+'/min.txt')
        max = np.loadtxt('../Data/UCI-MLR/'+client_data[client_id-1]+'/max.txt')

    benign_test = benign_data[-5000:]
    benign_test[np.isnan(benign_test)] = 0
    benign_test = (benign_test - min) / (max - min)

    benign_th = benign_data[5000:8000]
    benign_th[np.isnan(benign_th)] = 0
    benign_th = (benign_th - min) / (max - min)

    g_attack_data_list = [os.path.join('../Data/UCI-MLR/', client_data[client_id-1], 'gafgyt_attacks', f)
                              for f in os.listdir(os.path.join('../Data/UCI-MLR/', client_data[client_id-1], 'gafgyt_attacks'))]
    if client_data[client_id-1] == 'Ennio_Doorbell' or client_data[client_id-1] == 'Samsung_SNH_1011_N_Webcam':
        attack_data_list = g_attack_data_list
        benign_test = benign_test[-2500:]
    else:
        m_attack_data_list = [os.path.join('../Data/UCI-MLR/', client_data[client_id-1], 'mirai_attacks', f)
                                for f in os.listdir(os.path.join('../Data/UCI-MLR/', client_data[client_id-1], 'mirai_attacks'))]
        attack_data_list = g_attack_data_list + m_attack_data_list

    attack_data = pd.concat([pd.read_csv(f)[-500:] for f in attack_data_list])
    attack_data = np.array(attack_data)
    attack_data[np.isnan(attack_data)] = 0
    attack_data = (attack_data - min) / (max - min)

    train_data_local_dict = torch.utils.data.DataLoader(benign_test, batch_size=batch_size, shuffle=False, num_workers=0)
    test_data_local_dict = torch.utils.data.DataLoader(attack_data, batch_size=batch_size, shuffle=False, num_workers=0)
    th_local_dict = torch.utils.data.DataLoader(benign_th, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_data_local_dict, test_data_local_dict, th_local_dict


def load_model():
    model = AutoEncoder()
    model.load_state_dict(torch.load('model.ckpt', map_location=lambda storage, loc: storage))
    return model

def test(args, model, device, train_data_local_dict, test_data_local_dict, threshold):
    model.eval()
    true_negative = 0
    false_positive = 0
    true_positive = 0
    false_negative = 0

    thres_func = nn.MSELoss(reduction='none')

    # print("test benign data")
    train_data = train_data_local_dict
    for idx, inp in enumerate(train_data):
        # if idx >= round(len(train_data) * 2 / 3):
            inp = inp.to(device)
            diff = thres_func(model(inp), inp)
            mse = diff.mean(dim=1)
            false_positive += (mse > threshold).sum()
            true_negative += (mse <= threshold).sum()

    # print("test attack data")
    test_data = test_data_local_dict
    for idx, inp in enumerate(test_data):
        inp = inp.to(device)
        diff = thres_func(model(inp), inp)
        mse = diff.mean(dim=1)
        true_positive += (mse > threshold).sum()
        false_negative += (mse <= threshold).sum()
    
    # print("calculate acc")
    # accuracy = ((true_positive) + (true_negative)) \
    #             / ((true_positive) + (true_negative) + (false_positive) + (false_negative))
    # precision = (true_positive) / ((true_positive) + (false_positive))
    # false_positive_rate = (false_positive) / ((false_positive) + (true_negative))
    # tpr = (true_positive) / ((true_positive) + (false_negative))
    # tnr = (true_negative) / ((true_negative) + (false_positive))

    accuracy = torch.true_divide(((true_positive) + (true_negative)) \
                , ((true_positive) + (true_negative) + (false_positive) + (false_negative)))
    precision = torch.true_divide((true_positive) , ((true_positive) + (false_positive)))
    false_positive_rate = torch.true_divide((false_positive) , ((false_positive) + (true_negative)))
    tpr = torch.true_divide((true_positive) , ((true_positive) + (false_negative)))
    tnr = torch.true_divide((true_negative) , ((true_negative) + (false_positive)))

    print(accuracy, false_positive_rate, tpr, tnr)

    return accuracy, precision, false_positive_rate

if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # experimental result tracking
    # wandb.init(project='fediot', entity='automl', config=args)

    # PyTorch configuration
    torch.set_default_tensor_type(torch.DoubleTensor)

    device = torch.device("cpu")

    # load data
    train_data_local_dict, test_data_local_dict, th_local_dict = load_data(args.batch_size, args.client_id)

    # create model
    model = load_model()

    mse = list()
    thres_func = nn.MSELoss(reduction='none')
    train_data = th_local_dict
    for idx, inp in enumerate(train_data):
        inp = inp.to(device)
        diff = thres_func(model(inp), inp)
        mse.append(diff)

    mse_results_global = torch.cat(mse).mean(dim=1)
    
    threshold_global = torch.mean(mse_results_global) + 0 * torch.std(mse_results_global)
    test(args, model, device, train_data_local_dict, test_data_local_dict, threshold_global)
    print(threshold_global)

    threshold_global = torch.mean(mse_results_global) + 1 * torch.std(mse_results_global)
    test(args, model, device, train_data_local_dict, test_data_local_dict, threshold_global)
    print(threshold_global)

    threshold_global = torch.mean(mse_results_global) + 2 * torch.std(mse_results_global)
    test(args, model, device, train_data_local_dict, test_data_local_dict, threshold_global)
    print(threshold_global)

    threshold_global = torch.mean(mse_results_global) + 3 * torch.std(mse_results_global)
    test(args, model, device, train_data_local_dict, test_data_local_dict, threshold_global)
    print(threshold_global)
