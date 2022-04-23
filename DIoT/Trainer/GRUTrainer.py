import logging
import torch


class GRUTrainer():
    def __init__(self, model, args=None):
        self.model = model
        self.args = args

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
        loss_func = torch.nn.CrossEntropyLoss()     # use cross-entropy loss because of outputing probability
        s, s_predict = train_data
        # model training
        for idx, inp in enumerate(s):
            # inp = inp.to(device)
            optimizer.zero_grad()
            p = model(torch.Tensor([[inp]]))
            a = torch.LongTensor([s_predict[idx]])
            # print(p)
            # print(a)
            loss = loss_func(p, a)
            loss.backward()
            optimizer.step()

        return self.get_model_params()

    def save_model(self):
        torch.save(self.model.state_dict(), 'model.ckpt')
        return True