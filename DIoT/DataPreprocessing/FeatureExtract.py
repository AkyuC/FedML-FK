# Suppose each IP address is a class of IoT device
import dpkt
import socket
import pandas as pd
import numpy as np
import torch

def feature_extract(iot_ip_set:set, file_raw:str, file_save_feature:str, file_label:str="", file_save_label:str=""):
    # port_dict = dict()
    len_set = set()
    # read raw packets
    f_file_raw = open(file_raw, 'rb')
    try:
        packets_iter = dpkt.pcap.Reader(f_file_raw)
    except ValueError:
        f_file_raw = open(file_raw, 'rb')
        packets_iter = dpkt.pcapng.Reader(f_file_raw)
    if(file_label != ""):
        file_label_list = np.array(pd.read_csv(file_label))

    # feature extract
    last_timeslot = -1
    feature_list = list()
    feature_label_list = list()
    row_index = -1
    for ts, pkt in packets_iter:
        row_index += 1
        try:
            eth = dpkt.ethernet.Ethernet(pkt) #解包，物理层
            if not isinstance(eth.data, dpkt.ip.IP): #解包，网络层，判断网络层是否存在，
                continue
            ip = eth.data
            if not isinstance(ip.data, dpkt.tcp.TCP): #解包，判断传输层协议是否是TCP，即当你只需要TCP时，可用来过滤
                continue
        except Exception:
            pass
        transf_data = ip.data #传输层负载数据，基本上分析流量的人都是分析这部分数据，即应用层负载流量
        ip_src = socket.inet_ntoa(ip.src)
        ip_dst = socket.inet_ntoa(ip.dst)
        sport = transf_data.sport
        dport = transf_data.dport
        # Judgment and tag
        # c1 direction
        c1 = 0
        if(ip_src in iot_ip_set):
            c1 = 1
        elif(ip_dst in iot_ip_set):
            c1 = 0
        else:
            continue
        # c2 and c3 local and remote port type
        if(sport <= 1023):          # system port
            c2 = 0
            # if sport not in port_dict:
            #     port_dict[sport] = 1
            # else:
            #     port_dict[sport] += 1
        elif(sport <= 49151):       # user port
            c2 = 1
        else:                       # dynamic port
            c2 = 2
        if(dport <= 1023):          # system port
            c3 = 0
            # if dport not in port_dict:
            #     port_dict[dport] = 1
            # else:
            #     port_dict[dport] += 1
        elif(dport <= 49151):       # user port
            c3 = 1
        else:                       # dynamic port
            c3 = 2
        # c4 packeet length
        c4 = ip.len
        len_set.add(c4)
        # c5 TCP flags
        c5 = transf_data.flags
        # c6 encapsulated protocol types
        # 运行在TCP协议上的协议：HTTP、HTTPS、FTP、POP3、SMTP、Telnet、SSH和DNS，因此这里仅判断这8种类型和其他
        if(sport == 88 or dport == 88):         # HTTP
            c6 = 1
        elif(sport == 443 or dport == 443):     # HTTPS
            c6 = 2
        elif(sport == 20 or dport == 20 or sport == 21 or dport == 21):     # FTP
            c6 = 3
        elif(sport == 110 or dport == 110):     # POP3
            c6 = 4
        elif(sport == 465 or dport == 465):     # SMTP
            c6 = 5
        elif(sport == 23 or dport == 23):     # Telnet
            c6 = 6
        elif(sport == 22 or dport == 22):     # SSH
            c6 = 7
        elif(sport == 53 or dport == 53):     # DNS
            c6 = 8
        else:
            c6 = 9
        # c7 IAT bin
        if(last_timeslot == -1):        # init
            c7 = 0
        elif(ts - last_timeslot < 0.001/1000):  # less than 0.001 ms
            c7 = 1
        elif(ts - last_timeslot < 0.05/1000):  # less than 0.05 ms
            c7 = 2
        else:
            c7 = 3
        last_timeslot = ts

        feature = [c1, c2, c3, c4, c5, c6, c7]
        feature_list.append(feature)
        if(file_label != ""):
            feature_label_list.append(file_label_list[row_index][1])

    # file close    
    f_file_raw.close()

    # a = sorted(port_dict)
    # print(len_set)
    # print(len(len_set))

    # save feature data to file
    # name=["c1", "c2," "c3", "c4", "c5", "c6", "c7"]
    feature_list_tmp = np.array(feature_list)
    min = np.array(feature_list).min(axis=0)
    max = np.array(feature_list).max(axis=0)
    feature_list = (feature_list_tmp - min) / (max - min)

    # print("feature_list_len: "+str(len(feature_list)))
    # print("feature_label_list: "+str(len(feature_list)))
    pd.DataFrame(data=feature_list).to_csv(file_save_feature, index=False)
    if(file_label != ""):
        pd.DataFrame(data=feature_label_list).to_csv(file_save_label, index=False)

if __name__ == '__main__':
    iot_ip_set = set()
    id = "1.165"
    ip = "192.168." + id
    iot_ip_set.add(ip)
    file_raw = "../Data/18_06_09_bengin.pcap"
    file_save_feature = "../Data/18_06_09_bengin%s.csv" % id
    # file_label = "../Data/SYN DoS_labels.csv"
    # file_save_label = "../Data/SYN DoS_labels%s.csv" % id
    feature_extract(iot_ip_set, file_raw, file_save_feature)
    print("finished!")