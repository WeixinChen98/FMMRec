# coding: utf-8
# @email: enoche.chow@gmail.com

r"""
################################
"""

import os
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator


class AbstractBMMFTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config):
        self.config = config

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class BMMFTrainer(AbstractBMMFTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, fair_disc_dict = None, filters = None):
        super(BMMFTrainer, self).__init__(config)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.fair_disc_dict = None
        if fair_disc_dict is not None:
            self.fair_disc_dict = fair_disc_dict
            for idx in self.fair_disc_dict:
                self.fair_disc_dict[idx].optimizer = self._build_optimizer(self.fair_disc_dict[idx])
        
        if filters is not None:
            self.filters = filters
            self.filters.optimizer = self._build_optimizer(self.filters)

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None

        self.use_neg_sampling = config['use_neg_sampling']*1

        self.d_steps = config['d_steps']
        self.disc_reg_weight_filtered = config['disc_reg_weight_filtered']
        self.disc_reg_weight_biased = config['disc_reg_weight_biased']
        self.modality_choice =  config['modality'].lower()


    def _build_optimizer(self, model):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        # if model is None:
        #     model = self.model

        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return optimizer


    def _train_epoch(self, train_data, epoch_idx):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0, []
        total_loss = None
        loss_batches = []

        # print_flag = True
        for batch_idx, interaction in enumerate(train_data):
            user_ids = interaction[0]
            pos_item_ids = interaction[1]

            self.filters.optimizer.zero_grad()
            losses, biased_embedding, filtered_embedding = self.filters(user_ids)

            # if print_flag:
            #     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            #     print('reconstruction and reg: ', losses)

            for feat_idx in self.fair_disc_dict:
                # feat_idx from 1 to n
                discriminator = self.fair_disc_dict[feat_idx]
                biased_loss, filtered_loss = discriminator(biased_embedding, filtered_embedding, interaction[feat_idx + 1 + self.use_neg_sampling])


                # if print_flag:
                #     print(feat_idx)
                #     print('biased_loss: ', biased_loss)
                #     print('filtered_loss: ', filtered_loss)

                # self.logger.info('cosine loss: {}'.format(losses))
                # self.logger.info('disc loss: {}'.format(disc_losses))


                losses += (self.disc_reg_weight_biased * biased_loss - self.disc_reg_weight_filtered * filtered_loss)

            # print_flag = False

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()

            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)

            loss.backward()
            self.filters.optimizer.step()

            for feat_idx in self.fair_disc_dict:
                for _ in range(self.d_steps):
                    discriminator = self.fair_disc_dict[feat_idx]
                    discriminator.optimizer.zero_grad()
                    biased_loss, filtered_loss = discriminator(biased_embedding.detach(), filtered_embedding.detach(), interaction[feat_idx + 1 + self.use_neg_sampling])
                    disc_loss = biased_loss + filtered_loss
                    disc_loss.backward(retain_graph=False)
                    discriminator.optimizer.step()

            loss_batches.append(loss.detach())
        

            # for test
            #if batch_idx == 0:
            #    break
        return total_loss, loss_batches



    def _check_nan(self, loss):
        if torch.isnan(loss):
            #raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            #for param_group in self.model.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            
            # if (epoch_idx + 1) % 10 == 0:
            #     self.filters.save_modal_feature(epochs = epoch_idx + 1)

        self.filters.save_modal_feature(epochs = self.epochs)

        return 



