from .MqttComManager import MqttCommManager
from ComManager.BaseComManager.Message import Message


class ClientMqttCommManager(MqttCommManager):
    def __init__(self, host, port, id=0, topic="fediot"):
        """
            server_id is 0
            client_id ranges from 1 to N
        """
        super().__init__(host, port, id)
        self._topic = topic

    def _on_connect(self, client, userdata, flags, rc):
        """
            [server]
            sending message topic (publish): serverID_clientID
            receiving message topic (subscribe): clientID

            [client]
            sending message topic (publish): clientID
            receiving message topic (subscribe): serverID_clientID

        """
        print("client_id:{}, connection returned with result code:{}".format(self.id, rc))
        # client
        result, = self._client.subscribe(self._topic + str(0) + "_" + str(self.id), 0)
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
        # client
        self._client.publish(self._topic + str(self.id), payload=msg.to_json())