# 客户端选择（对抗学习）

### 文件功能：
<div align='center'>

|            文件名            |                          功能                          |
| :--------------------------: | :----------------------------------------------------: |
|       FedAvgServer.py        |                     联邦学习服务器                     |
|      FedAvgServer_CS.py      |  带客户端选择（基于局部模型测试精度）的联邦学习服务器  |
|     FedAvgServerTest.py      | 利用全部测试集，测试服务器在每个通信回合保存的模型精度 |
|       FedAvgClient.py        |                     联邦学习客户端                     |
|      FedAvgClient_CS.py      |           带局部模型精度测试的联邦学习客户端           |
|     FedAvgClientTest.py      |              利用局部测试集，测试模型精度              |
|  FedAvgClient_malicious.py   |                       恶意客户端                       |
| FedAvgClient_malicious_CS.py |             带局部模型精度测试的恶意客户端             |
</div>

### 实验设置：

对比实验①：无恶意客户端，5个客户端都参与每回合训练。

<div align='center'>

|        实验参数         |  参数值  |
| :---------------------: | :------: |
|       客户端数量        |    5     |
|  每回合选择客户端数量   |    5     |
|     恶意客户端数量      |    0     |
|     客户端学习率      |    1e-4 |
|     客户端选择策略      | 全部选择 |
|      模型聚合算法       |  FedAvg  |
| 通信回合数量 comm_round |    6     |
| 局部训练回合数量 epochs |    10    |
</div>

实验①结果：
![Acc](https://github.com/AkyuC/FedML-FK/blob/client_selection/FedAvg/Server/Acc.png)

---

对比实验②：添加2个恶意客户端，7个客户端都参与每回合训练。

由于恶意客户端的存在，训练相同回合数之后，模型测试精度应该比实验①差。

<div align='center'>

|        实验参数         |  参数值  |
| :---------------------: | :------: |
|       客户端数量        |    7     |
|  每回合选择客户端数量   |    7     |
|     恶意客户端数量      |    2     |
|     正常客户端学习率      |    1e-4     |
|     恶意客户端学习率      |    1e-3     |
|     客户端选择策略      | 全部选择 |
|      模型聚合算法       |  FedAvg  |
| 通信回合数量 comm_round |    6     |
| 局部训练回合数量 epochs |    10    |
</div>

实验②结果：
![Acc_malicious](https://github.com/AkyuC/FedML-FK/blob/client_selection/FedAvg/Server/Acc_malicious.png)

---

对比实验③：添加2个恶意客户端，7个客户端都参与每回合训练。

由于客户端选择机制，恶意客户端的影响将被削弱，模型测试精度应该比实验②好。

<div align='center'>

|        实验参数         |      参数值      |
| :---------------------: | :--------------: |
|       客户端数量        |        7         |
|  每回合选择客户端数量   |        7         |
|     恶意客户端数量      |        2         |
|     客户端选择策略      |     全部选择     |
|     正常客户端学习率      |    1e-4     |
|     恶意客户端学习率      |    1e-3     |
|      模型聚合算法       | 根据信誉分数加权 |
| 通信回合数量 comm_round |        6         |
| 局部训练回合数量 epochs |        10        |
</div>

实验③结果：
![Acc_CS](https://github.com/AkyuC/FedML-FK/blob/client_selection/FedAvg/Server/Acc_CS.png)
