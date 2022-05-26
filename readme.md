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
| 通信回合数量 comm_round |    15     |
| 局部训练回合数量 epochs |    10    |
</div>

实验①结果：全局模型的测试精度一直在上升，直到10次通信后基本收敛。
![Acc](https://github.com/AkyuC/FedML-FK/blob/client_selection/FedAvg/Server/Acc.png)

---

对比实验②：添加1个恶意客户端，6个客户端都参与每回合训练。

由于恶意客户端的存在，训练相同回合数之后，模型测试精度应该比实验①差。

<div align='center'>

|        实验参数         |  参数值  |
| :---------------------: | :------: |
|       客户端数量        |    6     |
|  每回合选择客户端数量   |    6     |
|     恶意客户端数量      |    1     |
|     正常客户端学习率      |    1e-4     |
|     恶意客户端学习率      |    1e-3     |
|     客户端选择策略      | 全部选择 |
|      模型聚合算法       |  FedAvg  |
| 通信回合数量 comm_round |    15     |
| 局部训练回合数量 epochs |    10    |
</div>

实验②结果：由于无法识别恶意客户端，导致全局模型的测试精度一直在下降。
![Acc_malicious](https://github.com/AkyuC/FedML-FK/blob/client_selection/FedAvg/Server/Acc_malicious.png)

---

对比实验③：无恶意客户端，5个客户端都参与每回合训练。

按照信誉分数计算公式
![](http://latex.codecogs.com/svg.latex?score=(1-\\tau)\\cdot(acc-new_{acc})+\\tau\\cdot(acc-old_{acc}))
，评估客户端的数据质量，从而进行客户端选择。

由于客户端选择机制，模型测试精度应该比实验①好。

<div align='center'>

|        实验参数         |      参数值      |
| :---------------------: | :--------------: |
|       客户端数量        |        5         |
|  每回合选择客户端数量   |        5         |
|     恶意客户端数量      |        0         |
|     客户端选择策略      |     全部选择     |
|     正常客户端学习率      |    1e-4     |
|      模型聚合算法       | 根据信誉分数加权 |
|   信誉分数软更新系数 tau |       0.5        |
| 通信回合数量 comm_round |        15         |
| 局部训练回合数量 epochs |        10        |
</div>

实验③结果：全局模型收敛之后，由于局部训练无法再提升模型精度，导致信誉分数评估不再准确，因此全局模型精度略有下降。
To do: 该方案并未明显体现**模型优化**的效果，考虑接下来尝试**偏向于选择“局部损失值较大”的客户端参与训练**的方案。
![Acc_CS](https://github.com/AkyuC/FedML-FK/blob/client_selection/FedAvg/Server/Acc_CS.png)

---

对比实验④：添加1个恶意客户端，6个客户端都参与每回合训练。

按照信誉分数计算公式
![](http://latex.codecogs.com/svg.latex?score=(1-\\tau)\\cdot(acc-new_{acc})+\\tau\\cdot(acc-old_{acc}))
，评估客户端的数据质量，从而进行客户端选择。

由于客户端选择机制，模型测试精度应该比实验②好。

<div align='center'>

|        实验参数         |      参数值      |
| :---------------------: | :--------------: |
|       客户端数量        |        6         |
|  每回合选择客户端数量   |        6         |
|     恶意客户端数量      |        1         |
|     客户端选择策略      |     全部选择     |
|     正常客户端学习率      |    1e-4     |
|     恶意客户端学习率      |    1e-3     |
|      模型聚合算法       | 根据信誉分数加权 |
|   信誉分数软更新系数 tau |       0.5        |
| 通信回合数量 comm_round |        15         |
| 局部训练回合数量 epochs |        10        |
</div>

实验④结果：效果并不理想，可能是由于全局模型聚合了恶意客户端上传的局部更新，导致信誉分数评估失效（无法体现局部模型和聚合后全局模型之间的差距），因此无法有效剔除恶意客户端的影响。
![Acc_malicious_CS](https://github.com/AkyuC/FedML-FK/blob/client_selection/FedAvg/Server/Acc_malicious_CS.png)
