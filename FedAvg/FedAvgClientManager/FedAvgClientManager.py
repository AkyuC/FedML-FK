import logging

from FedAvg.Utils.FedAvgManager import FedAvgManager
from FedAvg.Utils.FedAvgMessage import FedAvgMessage
from FedAvg.Utils.utils import transform_list_to_tensor, transform_tensor_to_list

from ComManager.BaseComManager.Message import Message


class FedAvgClientManager(FedAvgManager):
    def __init__(self, args, id, trainer, train_data_iter, train_data_num, device, HOST="192.168.10.188", PORT=1883, topic="fediot"):
        super().__init__(args, id=id, HOST=HOST, PORT=PORT, topic=topic)
        self.trainer = trainer
        self.device = device
        self.train_data_iter = train_data_iter
        self.train_data_num = train_data_num
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.is_finish = False

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(FedAvgMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(FedAvgMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(FedAvgMessage.MSG_ARG_KEY_MODEL_PARAMS)
        self.trainer.set_model_params(global_model_params)
        self.round_idx = 0
        self.__train()

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        print("handle_message_receive_model_from_server.")
        model_params = msg_params.get(FedAvgMessage.MSG_ARG_KEY_MODEL_PARAMS)
        model_params = transform_list_to_tensor(model_params)
        self.trainer.set_model_params(model_params)
        self.round_idx += 1
        self.__train()
        if self.round_idx == self.num_rounds - 1:
            print("finish and save model!")
            self.trainer.save_model(self.round_idx)
            self.finish()
            self.is_finish = True

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        print("send_model_to_server.")
        message = Message(FedAvgMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(FedAvgMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(FedAvgMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def __train(self):
        weights = self.trainer.train(self.train_data_iter, self.device, self.args)
        weights = transform_tensor_to_list(weights)
        self.send_model_to_server(0, weights, self.train_data_num)
