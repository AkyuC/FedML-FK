import logging
from statistics import mean
import torch
import torch.nn.functional as func
import matplotlib.pyplot as plt

class GRUTrainer():
    def __init__(self, model, args=None):
        self.model = model
        self.args = args
        self.out = torch.nn.Softmax(dim=0)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        logging.info(device)
        model = self.model.to(device)
        # model.double()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        s, s_predict = train_data
        loss_list = []
        # model training
        for idx, inp in enumerate(s):
            optimizer.zero_grad()
            inp = torch.Tensor([inp]).to(device)
            p = model(inp)
            p = self.out(p)
            a = torch.LongTensor(s_predict[idx])
            # use cross-entropy loss because of outputing probability
            loss = func.cross_entropy(p, a)
            loss_list.append(loss.item())
            # print(loss.item())
            loss.backward()
            optimizer.step()
        print("mean loss: " + str(mean(loss_list)))
        # plt.plot(loss_list)
        # plt.show()
        return self.get_model_params()

    def save_model(self):
        torch.save(self.model.state_dict(), 'model.ckpt')
        return True