import logging
import torch


class AETrainer():
    def __init__(self, model, args=None):
        self.model = model
        self.args = args

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data_iter, device, args):
        logging.info(device)
        model = self.model.to(device)
        model.double()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_func = torch.nn.MSELoss()
        # model training
        for epoch in range(args.epochs):
            # mini-batch loop
            for idx, inp in enumerate(train_data_iter):
                inp = inp.to(device)
                optimizer.zero_grad()
                decode = model(inp)
                loss = loss_func(decode, inp)
                loss.backward()
                optimizer.step()
        return self.get_model_params()

    def save_model(self):
        torch.save(self.model.state_dict(), 'model.ckpt')
        return True