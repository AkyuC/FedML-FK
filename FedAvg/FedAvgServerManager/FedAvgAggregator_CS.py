import logging
import time
import numpy as np
import math
import copy
import decimal

class FedAVGAggregator(object):

    def __init__(self, args, worker_num, device, model_trainer):
        self.trainer = model_trainer
        self.args = args
        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        self.global_acc_old = dict()
        self.global_acc = dict()
        self.random_acc = dict()
        self.model_owners = dict()
        self.score = dict()

        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num, global_acc, random_acc):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True
        if index in self.global_acc.keys():
            self.global_acc_old[index] = self.global_acc[index]
        else:
            self.global_acc_old[index] = 0
        self.global_acc[index] = global_acc
        self.random_acc[index] = random_acc
        # print("add local result")

    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        # print(self.worker_num)
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                # print("False")
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        # print("True")
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("aggregate len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # 计算客户端的信誉分数
        score = copy.deepcopy(self.score)
        # 前两个通信回合，初始化信誉分数都为0.1
        if 0 in self.global_acc_old.values():
            for index in range(self.worker_num):
                score[index] = 0.1
        else:
            # print("model_owners", self.model_owners)
            # print("random_acc", self.random_acc)
            # print("global_acc", self.global_acc)
            # print("global_acc_old", self.global_acc_old)
            for i in range(len(self.model_owners)):
                # 软更新系数为0.5
                tau = 0.5
                score[self.model_owners[i]] = score[self.model_owners[i]] * (1 - tau) + tau * \
                    (1.0 * (self.random_acc[i] - self.global_acc[i]) + 1.0 * (self.random_acc[i] - self.global_acc_old[i]))
                print("client", self.model_owners[i], "random_acc=", self.random_acc[i], "global_acc=", self.global_acc[i], \
                        "global_acc_old=", self.global_acc_old[i])
        print("score", score)
        self.score = copy.deepcopy(score)
        for i in range(self.worker_num):
            local_sample_number, local_model_params = model_list[i]
            # 超参数，底数为4
            score[i] = local_sample_number * math.pow(4, score[i])
        
        print("weight: ", end="")
        for i in range(len(score)):
            # print(decimal.Decimal(str(score[i])) / decimal.Decimal(str(sum(score.values()))))
            print(float(decimal.Decimal(str(score[i])) / decimal.Decimal(str(sum(score.values())))), end=" ")
        print()

        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                # w = local_sample_number / training_num
                # 基于信誉分数加权，高精度计算
                w = float(decimal.Decimal(str(score[i])) / decimal.Decimal(str(sum(score.values()))))
                if i == 0:
                    # print("i", i, "k", k)
                    # print(np.array(local_model_params[k]))
                    # print(w)
                    averaged_params[k] = np.array(local_model_params[k]) * w
                    # print(averaged_params[k])
                else:
                    # print("i", i, "k", k)
                    # print(np.array(local_model_params[k]))
                    # print(w)
                    averaged_params[k] += np.array(local_model_params[k]) * w
                    # print(averaged_params[k])
            averaged_params[k] = averaged_params[k].tolist()
        # print(averaged_params[k])

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))

        return averaged_params

    # 客户端选择模块，目前为随机采样固定数量的客户端参与训练
    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        # select client from the collection of feature matches
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
    
    # 随机匹配客户端和局部模型
    # model_owners(a)=b，表示客户端a 测试 由客户端b生成的局部模型
    def random_matching(self, round_idx, client_num_in_total):
        client_indexes = [client_index for client_index in range(client_num_in_total)]
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        # np.random.shuffle(client_indexes)
        # 为避免元素位置不发生改变，随机指定偏移量
        offset = np.random.randint(1, client_num_in_total-1)
        client_indexes = client_indexes[offset:] + client_indexes[:offset]
        # print("client_indexes", client_indexes)
        logging.info("model_owners = %s" % str(client_indexes))
        for index in range(client_num_in_total):
            self.model_owners[index] = client_indexes[index]
        print("model_owners", self.model_owners)
