import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# FMMR 
class BMMF_binary(nn.Module):
    def __init__(self, feature_info, config, dataloader):
        super().__init__()
        self.feature_info = feature_info
        self.dropout = config['attacker_dropout']
        self.neg_slope = config['neg_slope']
        self.criterion = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.out_dim = 1
        self.layers = config['attacker_layers']
        self.epochs = config['epochs']
        self.modality_choice =  config['modality'].lower()

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        if 'v' == self.modality_choice:
            feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
        elif 't' == self.modality_choice:
            feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
        elif 'a' == self.modality_choice:
            feat_file_path = os.path.join(dataset_path, config['audio_feature_file'])
        else:
            raise ValueError('wrong modality choice (out of [v, a, t])')

        if os.path.isfile(feat_file_path):
            self.feat = torch.from_numpy(np.load(feat_file_path, allow_pickle=True)).type(torch.FloatTensor).cuda()
        else:
            raise ValueError('not feat file')
        


        self.embed_dim = self.feat.shape[1]

        self.name = feature_info.name
        self.model_file_name = "{}_{}_{}".format(config['dataset'], feature_info.name, config['model'])
        if config['hyper_parameters']:
            for i in config['hyper_parameters']:
                self.model_file_name += '_{}={}'.format(i, config[i])
        self.model_file_name += '_variant.pt'
        self.model_path = os.path.join(config['model_path'], self.model_file_name)

        self.optimizer = None

        if self.layers == 1:
            self.biased_network = nn.Sequential(
                nn.Linear(int(self.embed_dim), self.out_dim, bias=True)
            )
        elif self.layers == 2:
            self.biased_network = nn.Sequential(
                nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
            )
        elif self.layers == 3:
            self.biased_network = nn.Sequential(
                nn.Linear(self.embed_dim, int(self.embed_dim / 2), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 2), int(self.embed_dim / 4), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 4), self.out_dim, bias=True)
            )
        else:
            logging.error('Valid layers ∈ \{1, 2, 3\}, invalid attacker layers: ' + self.layers)
            return

        if self.layers == 1:
            self.filtered_network = nn.Sequential(
                nn.Linear(int(self.embed_dim), self.out_dim, bias=True)
            )
        elif self.layers == 2:
            self.filtered_network = nn.Sequential(
                nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
            )
        elif self.layers == 3:
            self.filtered_network = nn.Sequential(
                nn.Linear(self.embed_dim, int(self.embed_dim / 2), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 2), int(self.embed_dim / 4), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 4), self.out_dim, bias=True)
            )
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

    def forward(self, biased_embedding, filtered_embedding, labels):
        biased_output = self.biased_predict(biased_embedding)['output']
        filtered_output = self.filtered_predict(filtered_embedding)['output']
        # random_labels = torch.randint_like(labels, low=0, high=1)
        if torch.cuda.device_count() > 0:
            labels = labels.cpu().type(torch.FloatTensor).cuda()
            # random_labels = random_labels.cpu().type(torch.FloatTensor).cuda()
        else:
            labels = labels.type(torch.FloatTensor)
            # random_labels = random_labels.type(torch.FloatTensor)
        biased_loss = self.criterion(biased_output.squeeze(), labels)
        filtered_loss = self.criterion(filtered_output.squeeze(), labels)
        return biased_loss, filtered_loss

    def biased_predict(self, embeddings):
        scores = self.biased_network(embeddings)
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


    def filtered_predict(self, embeddings):
        scores = self.filtered_network(embeddings)
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
    
    def save_modal_feature(self, file_path = './'):
        np.save(os.path.join(file_path, 'biased_{}_feat_epoch={}.npy'.format(self.modality_choice, self.epochs)), self.biased_trs(self.feat).detach().cpu())
        np.save(os.path.join(file_path, 'filtered_{}_feat_epoch={}.npy'.format(self.modality_choice, self.epochs)), self.filtered_trs(self.feat).detach().cpu())


class BMMF_multiclass(nn.Module):
    def __init__(self, feature_info, config, dataloader):
        super().__init__()
        self.feature_info = feature_info
        self.dropout = config['attacker_dropout']
        self.neg_slope = config['neg_slope']
        self.criterion = nn.NLLLoss()
        self.out_dim = feature_info.num_class

        self.layers = config['attacker_layers']
        self.epochs = config['epochs']
        self.modality_choice =  config['modality'].lower()

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        if 'v' == self.modality_choice:
            feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
        elif 't' == self.modality_choice:
            feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
        elif 'a' == self.modality_choice:
            feat_file_path = os.path.join(dataset_path, config['audio_feature_file'])
        else:
            raise ValueError('wrong modality choice (out of [v, a, t])')

        if os.path.isfile(feat_file_path):
            self.feat = torch.from_numpy(np.load(feat_file_path, allow_pickle=True)).type(torch.FloatTensor).cuda()
        else:
            raise ValueError('not feat file')
        

        self.embed_dim = self.feat.shape[1]


        self.name = feature_info.name
        self.model_file_name = "{}_{}_{}".format(config['dataset'], feature_info.name, config['model'])
        if config['hyper_parameters']:
            for i in config['hyper_parameters']:
                self.model_file_name += '_{}={}'.format(i, config[i])
        self.model_file_name += '_variant.pt'
        self.model_path = os.path.join(config['model_path'], self.model_file_name)

        self.optimizer = None

        if self.layers == 1:
            self.biased_network = nn.Sequential(
                nn.Linear(int(self.embed_dim), self.out_dim, bias=True)
            )
        elif self.layers == 2:
            self.biased_network = nn.Sequential(
                nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
            )
        elif self.layers == 3:
            self.biased_network = nn.Sequential(
                nn.Linear(self.embed_dim, int(self.embed_dim / 2), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 2), int(self.embed_dim / 4), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 4), self.out_dim, bias=True)
            )
        else:
            logging.error('Valid layers ∈ \{1, 2, 3\}, invalid attacker layers: ' + self.layers)
            return

        if self.layers == 1:
            self.filtered_network = nn.Sequential(
                nn.Linear(int(self.embed_dim), self.out_dim, bias=True)
            )
        elif self.layers == 2:
            self.filtered_network = nn.Sequential(
                nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
            )
        elif self.layers == 3:
            self.filtered_network = nn.Sequential(
                nn.Linear(self.embed_dim, int(self.embed_dim / 2), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 2), int(self.embed_dim / 4), bias=True),
                nn.LeakyReLU(self.neg_slope),
                nn.Dropout(p=self.dropout),
                nn.Linear(int(self.embed_dim / 4), self.out_dim, bias=True)
            )
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

    # TODO: using shared filtered and biased discriminators or not?
    def forward(self, biased_embedding, filtered_embedding, labels):

        biased_output = self.biased_predict(biased_embedding)['output']
        filtered_output = self.filtered_predict(filtered_embedding)['output']

        random_labels = torch.randint_like(labels, low=0, high=self.out_dim-1)

        biased_loss = self.criterion(biased_output.squeeze(), labels)
        filtered_loss = self.criterion(filtered_output.squeeze(), random_labels)

        return biased_loss, filtered_loss


    def biased_predict(self, embeddings):
        scores = self.biased_network(embeddings)
        output = F.log_softmax(scores, dim=1)
        prediction = output.max(1, keepdim=True)[1]     # get the index of the max
        result_dict = {'output': output,
                       'prediction': prediction}
        return result_dict

    def filtered_predict(self, embeddings):
        scores = self.filtered_network(embeddings)
        output = F.log_softmax(scores, dim=1)
        prediction = output.max(1, keepdim=True)[1]     # get the index of the max
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
    


