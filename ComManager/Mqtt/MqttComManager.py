import uuid
from typing import List
from abc import abstractmethod

import paho.mqtt.client as mqtt

from ComManager.BaseComManager.Message import Message
from ComManager.BaseComManager.Observer import Observer
from ComManager.BaseComManager.BaseComManager import BaseCommunicationManager


class MqttCommManager(BaseCommunicationManager):
    def __init__(self, host, port, id=-1, topic="fediot"):
        self._observers: List[Observer] = []
        # set client id
        self.id = id
        if id is None or id == -1:
            # ramdon generate a client id if dont set
            self.id = mqtt.base62(uuid.uuid4().int, padding=22)            

        # Construct a Client
        self._client = mqtt.Client(client_id=topic+'_'+str(self.id))
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.on_subscribe = self._on_subscribe

        # connect broker,connect() or connect_async()
        self._client.connect(host, port, keepalive=60)
        # call loop() and keep the connection with broker, re-connects automatically if disconnect
        self._client.loop_start()   

    def __del__(self):
        self._client.loop_stop()
        self._client.disconnect()

    @abstractmethod
    def _on_connect(self, client, userdata, flags, rc):
        pass

    @abstractmethod
    def send_message(self, msg: Message):
        pass

    @staticmethod
    def _on_disconnect(client, userdata, rc):
        print("--Disconnection returned result:" + str(rc))

    def _on_subscribe(self, client, userdata, mid, granted_qos):
        print("--onSubscribe :" + str(mid))

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def _on_message(self, client, userdata, msg):
        # Called when a message has been received on the topic
        msg.payload = str(msg.payload, encoding='utf-8')
        self._notify(str(msg.payload))

    def _notify(self, msg):
        msg_params = Message()
        msg_params.init_from_json_string(str(msg))
        msg_type = msg_params.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def handle_receive_message(self):
        pass

    def stop_receive_message(self):
        pass
