from ComManager.BaseComManager.Message import Message
from FedAvg.Utils.FedAvgManager import FedAvgManager
from FedAvg.Utils.FedAvgMessage import FedAvgMessage
from FedAvg.Utils.utils import transform_list_to_tensor
import copy


class FedAVGServerManager(FedAvgManager):
    def __init__(self, args, aggregator, id=0, client_num=0, HOST="192.168.10.186", PORT=1883, topic="fediot"):
        super().__init__(args, id=id, client_num=client_num, HOST=HOST, PORT=PORT, topic=topic)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.client_num = client_num
        self.round_idx = 0
        self.is_finish = False

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id - 1])

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(FedAvgMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(FedAvgMessage.MSG_ARG_KEY_SENDER)
        print("recived a update from client: {}".format(sender_id))
        model_params = msg_params.get(FedAvgMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(FedAvgMessage.MSG_ARG_KEY_NUM_SAMPLES)
        global_acc = msg_params.get(FedAvgMessage.MSG_ARG_KEY_GLOBAL_ACC)
        random_acc = msg_params.get(FedAvgMessage.MSG_ARG_KEY_RANDOM_ACC)

        # record every client model params
        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number, global_acc, random_acc)
        
        # 判断收到全部客户端的更新之后，开始模型聚合
        if self.aggregator.check_whether_all_receive():
            global_model_params = self.aggregator.aggregate()

            # start the next round
            self.round_idx += 1
            # 每轮通信过后都保存一次模型
            # 将模型参数转换为 tensor 格式，（深拷贝）保存模型
            self.aggregator.set_global_model_params(transform_list_to_tensor(copy.deepcopy(global_model_params)))
            self.aggregator.trainer.save_model(self.round_idx)

            if self.round_idx == self.round_num:
                # self.aggregator.set_global_model_params(transform_list_to_tensor(global_model_params))
                # self.aggregator.trainer.save_model()
                self.finish()
                self.is_finish = True
                return
            client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                                 self.args.client_num_per_round)
            
            self.aggregator.random_matching(self.round_idx, self.args.client_num_in_total)
            for receiver_id in range(1, self.client_num+1):
                # self.send_message_sync_model_to_client(receiver_id, global_model_params,
                #                                        client_indexes[receiver_id - 1])
                self.send_message_two_models_to_client(receiver_id, global_model_params, 
                    self.aggregator.model_dict[self.aggregator.model_owners[receiver_id-1]], client_indexes[receiver_id-1])


    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(FedAvgMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(FedAvgMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(FedAvgMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    # 服务器向客户端发送全局模型
    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        print("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(FedAvgMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(FedAvgMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(FedAvgMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)
    
    # 服务器向客户端发送全局模型+随机局部模型
    def send_message_two_models_to_client(self, receive_id, global_model_params, random_model_params, client_index):
        print("send_message_two_models_to_client. receive_id = %d" % receive_id)
        message = Message(FedAvgMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(FedAvgMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(FedAvgMessage.MSG_ARG_KEY_RANDOM_MODEL_PARAMS, random_model_params)
        message.add_params(FedAvgMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)
