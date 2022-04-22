import logging
import torch

from DIoT.DataPreprocessing.FeatureCluster import FeatureMapping


class GRUTrainer():
    def __init__(self, model, args=None):
        self.model = model
        self.args = args

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, k, kM, n_clusters, args):
        logging.info(device)
        model = self.model.to(device)
        model.double()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_func = torch.nn.CrossEntropyLoss()     # use cross-entropy loss because of outputing probability
        s = FeatureMapping(train_data, kM)
        train_data_len = len(train_data)
        # model training
        for index in range(train_data_len - k):
            optimizer.zero_grad()
            p = model(s[index:index+k])
            target = [0 for _ in range(n_clusters)]
            target[s[index+k]] = 1
            loss = loss_func(p, target)
            loss.backward()
            optimizer.step()
        return self.get_model_params()

    def save_model(self):
        torch.save(self.model.state_dict(), 'model.ckpt')
        return True