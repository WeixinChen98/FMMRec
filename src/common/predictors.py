import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


# FMMR
class BinaryPredictor(nn.Module):
    def __init__(self, feature_info, config):
        super().__init__()
        self.embed_dim = config['embedding_size']
        self.feature_info = feature_info
        self.dropout = config['attacker_dropout']
        self.neg_slope = config['neg_slope']
        self.criterion = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.out_dim = 1
        self.layers = config['attacker_layers']


        self.name = feature_info.name
        self.model_file_name = "{}_{}_{}".format(config['dataset'], feature_info.name, config['model'])
        for i in config['hyper_parameters']:
            self.model_file_name += '_{}={}'.format(i, config[i])
        self.model_file_name += '_pred.pt'
        self.model_path = os.path.join(config['model_path'], self.model_file_name)

        self.optimizer = None

        if self.layers == 1:
            self.network = nn.Sequential(
                nn.Linear(int(self.embed_dim), self.out_dim, bias=True)
            )
        elif self.layers == 2:
            self.network = nn.Sequential(
                nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
            )
        elif self.layers == 3:
            self.network = nn.Sequential(
                nn.Linear(self.embed_dim, int(self.embed_dim / 2), bias=True),
                # nn.BatchNorm1d(num_features=self.embed_dim * 4),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 2), int(self.embed_dim / 4), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 4), self.out_dim, bias=True)
            )
        # elif self.layers == 5:
        #     self.network = nn.Sequential(
        #         nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
        #         # nn.BatchNorm1d(num_features=self.embed_dim * 2),
        #         nn.LeakyReLU(self.neg_slope),
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
        #         nn.LeakyReLU(self.neg_slope),
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
        #         # nn.BatchNorm1d(num_features=self.embed_dim),
        #         nn.LeakyReLU(self.neg_slope),
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
        #         nn.LeakyReLU(self.neg_slope),
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        #         )
        # elif self.layers == 7:
        #     self.network = nn.Sequential(
        #         nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
        #         # nn.BatchNorm1d(num_features=self.embed_dim * 2),
        #         nn.LeakyReLU(self.neg_slope),
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
        #         # nn.BatchNorm1d(num_features=self.embed_dim * 4),
        #         nn.LeakyReLU(self.neg_slope),
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
        #         nn.LeakyReLU(self.neg_slope),
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
        #         nn.LeakyReLU(self.neg_slope),
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
        #         # nn.BatchNorm1d(num_features=self.embed_dim),
        #         nn.LeakyReLU(self.neg_slope),
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
        #         nn.LeakyReLU(self.neg_slope),
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        #        )
        else:
            logging.error('Valid layers ∈ \{1, 2, 3\}, invalid attacker layers: ' + self.layers)
            return


    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, embeddings, labels):
        output = self.predict(embeddings)['output']
        if torch.cuda.device_count() > 0:
            labels = labels.cpu().type(torch.FloatTensor).cuda()
        else:
            labels = labels.type(torch.FloatTensor)
        loss = self.criterion(output.squeeze(), labels)
        return loss

    def predict(self, embeddings):
        scores = self.network(embeddings)
        output = self.sigmoid(scores)

        # prediction = output.max(1, keepdim=True)[1]     # get the index of the max
        if torch.cuda.device_count() > 0:
            threshold = torch.tensor([0.5]).cuda()
        else:
            threshold = torch.tensor([0.5])
        prediction = (output > threshold).float() * 1

        result_dict = {'output': output,
                       'prediction': prediction}
        return result_dict

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        # logging.info('Save ' + self.name + ' discriminator model to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        logging.info('Load ' + self.name + ' discriminator model from ' + model_path)