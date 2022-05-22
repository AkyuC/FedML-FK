import logging
import time
import numpy as np
import math

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
        self.global_acc_old[index] = self.global_acc[index]
        self.global_acc[index] = global_acc
        self.random_acc[index] = random_acc

    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
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
        score = dict()
        # 首个通信回合，初始化信誉分数都为0.1
        if len(self.global_acc_old) == 0:
            for index in range(self.worker_num):
                score[index] = 0.1
        else:
            for i in range(len(self.model_owners)):
                # 软更新系数为0.2
                score[self.model_owners[i]] = score[self.model_owners[i]] * (1 - 0.2) + 0.2 * \
                    (1.0 * (self.random_acc[i] - self.global_acc[i]) + 1.0 * (self.random_acc[i] - self.global_acc_old[i]))
        
        for i in range(self.worker_num):
            local_sample_number, local_model_params = model_list[i]
            # 超参数，底数为4
            score[i] = local_sample_number * math.pow(4, score[i])

        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                # local_sample_number, local_model_params = model_list[i]
                # w = local_sample_number / training_num
                # 基于信誉分数加权
                w = score[i] / sum(score.values())
                if i == 0:
                    averaged_params[k] = np.array(local_model_params[k]) * w
                else:
                    averaged_params[k] += np.array(local_model_params[k]) * w
            averaged_params[k] = averaged_params[k].tolist()

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
        model_owners = np.random.shuffle(client_indexes)
        logging.info("model_owners = %s" % str(model_owners))
        for index in range(client_num_in_total):
            self.model_owners[index] = model_owners[index]
