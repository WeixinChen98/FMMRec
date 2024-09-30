import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# FMMR 
class BMMF_filters(nn.Module):
    def __init__(self, config, dataloader):
        super().__init__()
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
        

        self.interaction_matrix = dataloader.inter_matrix(form='coo').astype(np.float32)
        self.interaction_matrix = self.sparse_mx_to_torch_sparse_tensor(self.interaction_matrix).float().to_dense()
        row_sums = self.interaction_matrix.sum(axis=-1)
        self.interaction_matrix = self.interaction_matrix / row_sums[:, np.newaxis]
        self.interaction_matrix = self.interaction_matrix.cuda()

        self.embed_dim = self.feat.shape[1]

        print('self.embed_dim', self.embed_dim)
        print(feat_file_path)

        self.model_file_name = "{}_{}".format(config['dataset'], config['model'])
        if config['hyper_parameters']:
            for i in config['hyper_parameters']:
                self.model_file_name += '_{}={}'.format(i, config[i])
        self.model_file_name += '_BMMFfilters.pt'
        self.model_path = os.path.join(config['model_path'], self.model_file_name)

        self.optimizer = None


        self.biased_trs = nn.Linear(self.feat.shape[1], self.feat.shape[1])
        self.filtered_trs = nn.Linear(self.feat.shape[1], self.feat.shape[1])

        # self.biased_trs = nn.Sequential(
        #         nn.Linear(self.feat.shape[1], self.feat.shape[1]),
        #         nn.LeakyReLU(0.2, inplace=True),
        #         nn.Linear(self.feat.shape[1], self.feat.shape[1]),
        # )
        # self.filtered_trs = nn.Sequential(
        #         nn.Linear(self.feat.shape[1], self.feat.shape[1]),
        #         nn.LeakyReLU(0.2, inplace=True),
        #         nn.Linear(self.feat.shape[1], self.feat.shape[1]),
        # )
        self.cos_loss = nn.CosineEmbeddingLoss()

        self.dataset_path = dataset_path


    def forward(self, users):
        user_modal_feats = torch.mm(self.interaction_matrix[users], self.feat)
        biased_embedding = self.biased_trs(user_modal_feats)
        filtered_embedding = self.filtered_trs(user_modal_feats)
        target = torch.Tensor([1]).cuda()
        # biased_reconstruction_loss = self.cos_loss(biased_embedding, user_modal_feats, target)
        filtered_reconstruction_loss = self.cos_loss(filtered_embedding, user_modal_feats, target)

        target = torch.Tensor([-1]).cuda()
        reg_loss = self.cos_loss(biased_embedding, filtered_embedding, target)
        # return biased_reconstruction_loss + filtered_reconstruction_loss, biased_embedding, filtered_embedding
        return filtered_reconstruction_loss + 0.1 * reg_loss, biased_embedding, filtered_embedding
        # return filtered_reconstruction_loss, biased_embedding, filtered_embedding

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
    
    def save_modal_feature(self, epochs = None, file_path = None):
        if file_path is None:
            file_path = self.dataset_path
        if epochs is None:
            epochs = self.epochs
        np.save(os.path.join(file_path, 'biased_{}_feat_epoch={}.npy'.format(self.modality_choice, epochs)), self.biased_trs(self.feat).detach().cpu())
        np.save(os.path.join(file_path, 'filtered_{}_feat_epoch={}.npy'.format(self.modality_choice, epochs)), self.filtered_trs(self.feat).detach().cpu())

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
