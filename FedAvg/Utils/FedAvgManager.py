from cgitb import Hook
import logging
from abc import abstractmethod


from ComManager.BaseComManager.Message import Message
from ComManager.BaseComManager.Observer import Observer
from ComManager.Mqtt.ClientMqttComManager import ClientMqttCommManager
from ComManager.Mqtt.ServerMqttComManager import ServerMqttCommManager


class FedAvgManager(Observer):
    def __init__(self, args, id=0, client_num=0, HOST="192.168.10.188", PORT=1883, topic="fediot"):
        self.args = args
        self.id = id
        if id == 0:
            self.com_manager = ServerMqttCommManager(HOST, PORT, id, client_num, topic=topic)
        else:
            self.com_manager = ClientMqttCommManager(HOST, PORT, id=id, topic=topic)
        self.com_manager.add_observer(self)
        self.message_handler_dict = dict()

    def run(self):
        self.register_message_receive_handlers()
        self.com_manager.handle_receive_message()

    def get_sender_id(self):
        return self.id

    def receive_message(self, msg_type, msg_params) -> None:
        handler_callback_func = self.message_handler_dict[msg_type]
        handler_callback_func(msg_params)

    def send_message(self, message):
        self.com_manager.send_message(message)

    @abstractmethod
    def register_message_receive_handlers(self) -> None:
        pass

    def register_message_receive_handler(self, msg_type, handler_callback_func):
        self.message_handler_dict[msg_type] = handler_callback_func

    def finish(self):
        logging.info("__finish client")
        self.com_manager.stop_receive_message()
