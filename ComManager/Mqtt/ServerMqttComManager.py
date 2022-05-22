import logging

from .MqttComManager import MqttCommManager
from ComManager.BaseComManager.Message import Message


class ServerMqttCommManager(MqttCommManager):
    def __init__(self, host, port, id=0, client_num=0, topic="fediot"):
        """
            server_id is 0 
            client_id ranges from 1 to N
        """
        super().__init__(host, port, id)
        self.client_num = client_num
        self._topic = topic
        # record the subscribe client
        self.sub_client = list()

    def _on_connect(self, client, userdata, flags, rc):
        """
            [server]
            sending message topic (publish): serverID_clientID
            receiving message topic (subscribe): clientID

            [client]
            sending message topic (publish): clientID
            receiving message topic (subscribe): serverID_clientID

        """
        print("server_id:{}, connection returned with result code:{}".format(self.id, rc))
        # server
        for client_ID in range(1, self.client_num+1):
            result, mid = self._client.subscribe(self._topic + str(client_ID), 0)
            self.sub_client.append(mid)
            print(result)

    def send_message(self, msg: Message):
        """
            [server]
            sending message topic (publish): serverID_clientID
            receiving message topic (subscribe): clientID

            [client]
            sending message topic (publish): clientID
            receiving message topic (subscribe): serverID_clientID

        """
        # server
        receiver_id = msg.get_receiver_id()
        topic = self._topic + str(0) + "_" + str(receiver_id)
        logging.info("topic = %s" % str(topic))
        payload = msg.to_json()
        self._client.publish(topic, payload=payload)
        # self._client.publish(topic, payload=msg.to_json())
        logging.info("sent")