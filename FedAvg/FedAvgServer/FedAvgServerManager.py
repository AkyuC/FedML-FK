from ComManager.BaseComManager.Message import Message
from Utils.FedAvgManager import FedAvgManager
from Utils.FedAvgMessage import FedAvgMessage
from Utils.utils import transform_list_to_tensor


class FedAVGServerManager(FedAvgManager):
    def __init__(self, args, aggregator, id=0, client_num=0, HOST="192.168.10.188", PORT=1883):
        super().__init__(args, id=id, client_num=client_num, HOST=HOST, PORT=PORT)
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

        # record every client model params
        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        
        if self.aggregator.check_whether_all_receive():
            global_model_params = self.aggregator.aggregate()

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                self.aggregator.set_global_model_params(transform_list_to_tensor(global_model_params))
                self.aggregator.trainer.save_model()
                self.finish()
                self.is_finish = True
                return
            client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                                 self.args.client_num_per_round)

            for receiver_id in range(1, self.client_num+1):
                self.send_message_sync_model_to_client(receiver_id, global_model_params,
                                                       client_indexes[receiver_id - 1])

    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(FedAvgMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(FedAvgMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(FedAvgMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        print("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(FedAvgMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(FedAvgMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(FedAvgMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)
