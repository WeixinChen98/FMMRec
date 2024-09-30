# coding=utf-8
from utils.metrics import *
from sklearn.metrics import *
from sklearn.preprocessing import LabelBinarizer
import itertools as it
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity, smooth_l1_loss
from time import time
import numpy as np
import pandas as pd
import gc
import os
import torch
from logging import getLogger


def format_metric(metric):
    if type(metric) is not tuple and type(metric) is not list:
        metric = [metric]
    format_str = []
    if type(metric) is tuple or type(metric) is list:
        for m in metric:
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('%.4f' % m)
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('%d' % m)
    return ','.join(format_str)


def batch_to_gpu(batch):
    if torch.cuda.device_count() > 0:
        for c in batch:
            if type(batch[c]) is torch.Tensor:
                batch[c] = batch[c].cuda()
    return batch

class DiscriminatorTrainer:
    def __init__(self, optimizer='GD', learning_rate=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, check_epoch=10, early_stop=1, num_worker=0, disc_epoch=1000):
        """
        初始化
        :param optimizer: optimizer name
        :param learning_rate: learning rate
        :param epoch: total training epochs
        :param batch_size: batch size for training
        :param eval_batch_size: batch size for evaluation
        :param dropout: dropout rate
        :param l2: l2 weight
        :param metrics: evaluation metrics list
        :param check_epoch: check intermediate results in every n epochs
        :param early_stop: 1 for early stop, 0 for not.
        :param disc_epoch: number of epoch for training extra discriminator
        """
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout = dropout
        self.no_dropout = 0.0
        self.l2_weight = l2
        self.disc_epoch = disc_epoch

        self.check_epoch = check_epoch
        self.early_stop = early_stop
        self.time = None

        # record train, validation, test results
        self.train_results, self.valid_results = [], []
        self.pred_results, self.disc_results = [], []

        self.num_worker = num_worker
        self.logger = getLogger()

    def _build_optimizer(self, model, lr=None, l2_weight=None):
        optimizer_name = self.optimizer_name.lower()
        if lr is None:
            lr = self.learning_rate
        if l2_weight is None:
            l2_weight = self.l2_weight

        if optimizer_name == 'gd':
            self.logger.info("Optimizer: GD")
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer_name == 'adagrad':
            self.logger.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(
                model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer_name == 'adam':
            self.logger.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=l2_weight)
        else:
            self.logger.error("Unknown Optimizer: " + self.optimizer_name)
            assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, weight_decay=l2_weight)
        return optimizer

    @staticmethod
    def _get_masked_disc(disc_dict, labels, mask):
        if np.sum(mask) == 0:
            return []
        masked_disc_label = [(disc_dict[i + 1], labels[:, i]) for i, val in enumerate(mask) if val != 0]
        return masked_disc_label


    @torch.no_grad()
    def _eval_discriminator(self, model, labels, u_vectors, fair_disc_dict, num_disc):
        feature_info = model.data_processor_dict['train'].data_reader.feature_info
        feature_eval_dict = {}
        for i in range(num_disc):
            discriminator = fair_disc_dict[i + 1]
            label = labels[:, i]
            feature_name = feature_info[i + 1].name
            discriminator.eval()
            if feature_info[i + 1].num_class == 2:
                prediction = discriminator.predict(u_vectors)['prediction'].squeeze()
            else:
                prediction = discriminator.predict(u_vectors)['output']
            feature_eval_dict[feature_name] = {'label': label.cpu(), 'prediction': prediction.detach().cpu(),
                                               'num_class': feature_info[i + 1].num_class}
            discriminator.train()
        return feature_eval_dict

    @staticmethod
    def _disc_eval_method(label, prediction, num_class):
        if num_class == 2:
            score = roc_auc_score(label, prediction, average='micro')
            score = max(score, 1 - score)
        else:
            score = f1_score(label, prediction, average='micro')
  
        return score

    def check(self, model, out_dict):
        """
        Check intermediate results
        :param model: model obj
        :param out_dict: output dictionary
        :return:
        """
        check = out_dict
        self.logger.info(os.linesep)
        for i, t in enumerate(check['check']):
            d = np.array(t[1].detach().cpu())
            self.logger.info(os.linesep.join(
                [t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss, l2 = check['loss'], model.l2()
        l2 = l2 * self.l2_weight
        l2 = l2.detach()
        self.logger.info('loss = %.4f, l2 = %.4f' % (loss, l2))

    def train_discriminator(self, model, dp_dict, fair_disc_dict, lr_attack=None, l2_attack=None, from_item = False):
        """
        Train discriminator to evaluate the quality of learned embeddings
        :param model: trained model
        :param dp_dict: Data processors for train valid and test
        :param fair_disc_dict: fairness discriminator dictionary
        :return:
        """
        train_data = DataLoader(dp_dict['train'], batch_size=dp_dict['train'].batch_size, num_workers=self.num_worker,
                                shuffle=True, collate_fn=dp_dict['train'].collate_fn)
        test_data = DataLoader(dp_dict['test'], batch_size=dp_dict['test'].batch_size, num_workers=self.num_worker,
                               pin_memory=True, collate_fn=dp_dict['test'].collate_fn)

        feature_results = defaultdict(list)
        best_results = dict()

        try:
            for epoch in range(self.disc_epoch):
                test_result_dict = \
                    self.evaluation_disc(model, fair_disc_dict, test_data, dp_dict['train'], from_item = from_item)
                d_score_dict = test_result_dict['d_score']

                training_time = self._check_time()
                self.logger.info("Epoch %5d [%.1f s]" % (epoch + 1, training_time))

                for f_name in d_score_dict:
                    self.logger.info("%s disc test = %s" % (f_name, format_metric(d_score_dict[f_name])))
                    feature_results[f_name].append(d_score_dict[f_name])
                    if d_score_dict[f_name] == max(feature_results[f_name]):
                        best_results[f_name] = d_score_dict[f_name]
                        idx = dp_dict['train'].data_reader.f_name_2_idx[f_name]
                        fair_disc_dict[idx].save_model()

        except KeyboardInterrupt:
            self.logger.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                for idx in fair_disc_dict:
                    fair_disc_dict[idx].save_model()

        for f_name in best_results:
            self.logger.info("{} best: {:.4f}".format(
                f_name, best_results[f_name]))

        for idx in fair_disc_dict:
            fair_disc_dict[idx].load_model()
        
        return best_results

    def fit_disc(self, model, batches, fair_disc_dict, epoch=-1, lr_attack=None, l2_attack=None, from_item = False):
        """
        Train the discriminator
        :param model: model instance
        :param batches: train data in batches
        :param fair_disc_dict: fairness discriminator dictionary
        :param epoch: epoch number
        :param lr_attack: attacker learning rate
        :param l2_attack: l2 regularization weight for attacker
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        for idx in fair_disc_dict:
            discriminator = fair_disc_dict[idx]
            if discriminator.optimizer is None:
                discriminator.optimizer = self._build_optimizer(
                    discriminator, lr=lr_attack, l2_weight=l2_attack)
            discriminator.train()

        output_dict = dict()
        loss_acc = defaultdict(list)

        # Run for FMMR and FMMR4 only
        model.get_fair_representation()

        for batch in batches:
            mask = [1] * len(fair_disc_dict)
            mask = np.asarray(mask)

            batch = batch_to_gpu(batch)

            labels = batch['features']
            masked_disc_label = self._get_masked_disc(fair_disc_dict, labels, mask)

            # calculate recommendation loss + fair discriminator penalty
            uids = batch['X']
            if not from_item:
                vectors = model.get_user_embedding(uids)
            else:
                vectors = model.get_user_aggregated_item_embedding(uids)
            output_dict['check'] = []

            # update discriminator
            if len(masked_disc_label) != 0:
                for idx, (discriminator, label) in enumerate(masked_disc_label):
                    discriminator.optimizer.zero_grad()
                    disc_loss = discriminator(vectors.detach(), label)
                    disc_loss.backward()
                    discriminator.optimizer.step()
                    loss_acc[discriminator.name].append(disc_loss.detach().cpu())

        for key in loss_acc:
            loss_acc[key] = np.mean(loss_acc[key])

        output_dict['loss'] = loss_acc
        return output_dict

    @torch.no_grad()
    def evaluation_disc(self, model, fair_disc_dict, test_data, dp, from_item = False):
        num_features = dp.data_reader.num_features

        def eval_disc(labels, u_vectors, fair_disc_dict, num_features):
            feature_info = dp.data_reader.feature_info
            feature_eval_dict = {}
            for i in range(num_features):
                discriminator = fair_disc_dict[i + 1]
                label = labels[:, i]
                feature_name = feature_info[i + 1].name
                discriminator.eval()

                prediction = discriminator.predict(u_vectors)['prediction'].squeeze()
                prediction = torch.randint(low = 0, high = feature_info[i + 1].num_class - 1, size = prediction.size())
                feature_eval_dict[feature_name] = {'label': label.cpu(), 'prediction': prediction.detach().cpu(),
                                                   'num_class': feature_info[i + 1].num_class}
                discriminator.train()
            return feature_eval_dict

        eval_dict = {}

        for batch in test_data:

            batch = batch_to_gpu(batch)

            labels = batch['features']
            uids = batch['X'] 
            if not from_item:
                vectors = model.get_user_embedding(uids)
            else:
                vectors = model.get_user_aggregated_item_embedding(uids)

            batch_eval_dict = eval_disc(labels, vectors.detach(), fair_disc_dict, num_features=num_features)
            for f_name in batch_eval_dict:
                if f_name not in eval_dict:
                    eval_dict[f_name] = batch_eval_dict[f_name]
                else:
                    new_label = batch_eval_dict[f_name]['label']
                    current_label = eval_dict[f_name]['label']
                    eval_dict[f_name]['label'] = torch.cat((current_label, new_label), dim=0)

                    new_prediction = batch_eval_dict[f_name]['prediction']
                    current_prediction = eval_dict[f_name]['prediction']
                    eval_dict[f_name]['prediction'] = torch.cat((current_prediction, new_prediction), dim=0)

        # generate discriminator evaluation scores
        d_score_dict = {}
        if eval_dict is not None:
            for f_name in eval_dict:
                l = eval_dict[f_name]['label']
                pred = eval_dict[f_name]['prediction']
                n_class = eval_dict[f_name]['num_class']
                d_score_dict[f_name] = self._disc_eval_method(l, pred, n_class)

        output_dict = dict()
        output_dict['d_score'] = d_score_dict
        return output_dict



    def check_disc(self, out_dict):
        check = out_dict
        self.logger.info(os.linesep)
        for i, t in enumerate(check['check']):
            d = np.array(t[1].detach().cpu())
            self.logger.info(os.linesep.join(
                [t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss_dict = check['loss']
        for disc_name, disc_loss in loss_dict.items():
            self.logger.info('%s loss = %.4f' % (disc_name, disc_loss))

        # for discriminator
        if 'd_score' in out_dict:
            disc_score_dict = out_dict['d_score']
            for feature in disc_score_dict:
                self.logger.info('{} AUC = {:.4f}'.format(
                    feature, disc_score_dict[feature]))

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time