# coding: utf-8
# @email  : enoche.chow@gmail.com

import os
import numpy as np
import torch
import torch.nn as nn
from logging import getLogger
import scipy.sparse as sp
from time import gmtime, strftime

class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError
    #
    # def __str__(self):
    #     """
    #     Model prints with number of trainable parameters
    #     """
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']


        # load logger
        self.logger = getLogger()

        # load model path
        self.model_path = "{}{}_{}_{}_vision={}_text={}_audio={}_discWeight={}_".format(config['model_path'], config['dataset'], config['recommendation_model'], config['fairness_model'], config['vision_feature_file'], config['text_feature_file'], config['audio_feature_file'], config['disc_reg_weight'])
        for i in config['hyper_parameters']:
            self.model_path += '_{}={}'.format(i, config[i])
        current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        self.model_path += current_time + '.pt'

        self.logger.info(self.model_path)


        # load encoded features here
        self.v_feat, self.t_feat, self.a_feat = None, None, None
        if not config['end2end'] and config['is_multimodal_model']:
            self.dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            # if file exist?
            
            
            if config['vision_feature_file'] is not None:
                v_feat_file_path = os.path.join(self.dataset_path, config['vision_feature_file'])
                if os.path.isfile(v_feat_file_path):
                    self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)
            if  config['text_feature_file'] is not None:
                t_feat_file_path = os.path.join(self.dataset_path, config['text_feature_file'])
                if os.path.isfile(t_feat_file_path):
                    self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)
            if  config['audio_feature_file'] is not None:    
                a_feat_file_path = os.path.join(self.dataset_path, config['audio_feature_file'])
                if os.path.isfile(a_feat_file_path):
                    self.a_feat = torch.from_numpy(np.load(a_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)

            assert self.v_feat is not None or self.t_feat is not None or  self.a_feat is not None, 'Features all NONE'

        
    
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        self.logger.info('Save model to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        self.logger.info('Load model from ' + model_path)

    def freeze_model(self):
        self.eval()
        for params in self.parameters():
            params.requires_grad = False


    def get_fair_representation(self):
        return 