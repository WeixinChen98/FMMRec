# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset, DiscriminatorDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader, DiscriminatorDataReader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_recommendation_model, get_trainer, dict2str, get_disc_trainer
import platform
import os

# FMMR
from collections import defaultdict
from torch.utils.data import DataLoader
from common.discriminators import BinaryDiscriminator, MulticlassDiscriminator
import torch

import argparse


# scheme 2
from BMMF_trainer import BMMFTrainer
from BMMF import BMMF_binary, BMMF_multiclass
from BMMF_filters import BMMF_filters

def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    # logger.info('\n====Validation====\n' + str(valid_dataset))
    # logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    # (valid_data, test_data) = (
    #     EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
    #     EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    init_seed(config['seed'])

    # set random state of dataloader
    train_data.pretrain_setup()


    # create data reader
    disc_data_reader = DiscriminatorDataReader(path=config['data_path'], dataset_name=config['dataset'], feature_columns=config['feature_columns'], sep='\t', test_ratio=0.2)


    filters = BMMF_filters(config, train_data).cuda()


    # create discriminators
    fair_disc_dict = {}
    for feat_idx in disc_data_reader.feature_info:
        if disc_data_reader.feature_info[feat_idx].num_class == 2:
            fair_disc_dict[feat_idx] = \
                BMMF_binary(disc_data_reader.feature_info[feat_idx], config, train_data)
        else:
            fair_disc_dict[feat_idx] = \
                BMMF_multiclass(disc_data_reader.feature_info[feat_idx], config, train_data)
        fair_disc_dict[feat_idx].apply(fair_disc_dict[feat_idx].init_weights)
        if torch.cuda.device_count() > 0:
            fair_disc_dict[feat_idx] = fair_disc_dict[feat_idx].cuda()

    # trainer loading and initialization
    
    trainer = BMMFTrainer(config, fair_disc_dict = fair_disc_dict, filters = filters)
    # debug
    # model training
    trainer.fit(train_data, saved=save_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recommendation_model', '-m', type=str, default='VBPR', help='name of recommendation models')
    parser.add_argument('--dataset', '-d', type=str, default='microlens', help='name of datasets')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='gpu_id')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='epoch')
    parser.add_argument('--d_steps', '-ds', type=int, default=1, help='discriminator update steps')
    # parser.add_argument('--disc_reg_weight', type=float, default=0.1)
    parser.add_argument('--disc_reg_weight_biased', type=float, default=0.1)
    parser.add_argument('--disc_reg_weight_filtered', type=float, default=0.1)
    parser.add_argument('--modality', type=str, default='v')

    args = parser.parse_args()

    config_dict = {
        'gpu_id': args.gpu_id,
        'modality': args.modality,
        'fairness_model': None,
        'epochs':args.epochs,
        # 'disc_reg_weight': args.disc_reg_weight,
        'disc_reg_weight_biased': args.disc_reg_weight_biased,
        'disc_reg_weight_filtered': args.disc_reg_weight_filtered,
        'd_steps': args.d_steps
    }

    quick_start(model = args.recommendation_model, dataset = args.dataset, config_dict = config_dict)


# nohup python -u BMMF_runner.py --dataset microlens --modality t --gpu_id 0 > ../results/microlens/BMMF_t.log 2>&1 &
# nohup python -u BMMF_runner.py --dataset microlens --modality v --gpu_id 1 > ../results/microlens/BMMF_v.log 2>&1 &
# nohup python -u BMMF_runner.py --dataset microlens --modality a --gpu_id 1 > ../results/microlens/BMMF_a.log 2>&1 &

# shared gender clf, independent age/occ clf
# nohup python -u BMMF_runner.py --dataset ml1m --modality t --gpu_id 0 > ../results/ml1m/BMMF_t.log 2>&1 &
# nohup python -u BMMF_runner.py --dataset ml1m --modality v --gpu_id 0 > ../results/ml1m/BMMF_v.log 2>&1 &
# nohup python -u BMMF_runner.py --dataset ml1m --modality a --gpu_id 0 > ../results/ml1m/BMMF_a.log 2>&1 &


# nohup python -u BMMF_runner.py --dataset ml1m --modality v --gpu_id 0 --disc_reg_weight_biased 0.5 > ../results/ml1m/BMMF_v_bw=0.5.log 2>&1 &
# nohup python -u BMMF_runner.py --dataset ml1m --modality a --gpu_id 1 --disc_reg_weight_biased 0.5 > ../results/ml1m/BMMF_a_bw=0.5.log 2>&1 &


# nohup python -u BMMF_runner.py --dataset ml1m --modality v --gpu_id 0 --d_steps 10 > ../results/ml1m/BMMF_v_ds=10.log 2>&1 &
# nohup python -u BMMF_runner.py --dataset ml1m --modality a --gpu_id 1 --d_steps 10 > ../results/ml1m/BMMF_a_ds=10.log 2>&1 &



# independent classifiers
# nohup python -u BMMF_runner.py --dataset ml1m --modality t --gpu_id 0 > ../results/ml1m/BMMF_t.log 2>&1 &
# nohup python -u BMMF_runner.py --dataset ml1m --modality v --gpu_id 1 > ../results/ml1m/BMMF_v.log 2>&1 &
# nohup python -u BMMF_runner.py --dataset ml1m --modality a --gpu_id 2 > ../results/ml1m/BMMF_a.log 2>&1 &


# nohup python -u BMMF_runner.py --dataset ml1m --modality v --gpu_id 0 --d_steps 10 --disc_reg_weight_filtered 0.01 > ../results/ml1m/BMMF_v_ds=10_fw=0.01.log 2>&1 &

# nohup python -u BMMF_runner.py --dataset ml1m --modality v --gpu_id 0 --d_steps 10 --disc_reg_weight_filtered 0.0 > ../results/ml1m/BMMF_v_ds=10_fw=0.0.log 2>&1 &


# quick_start(model = 'VBPR', dataset = 'microlens', config_dict = {'gpu_id': 0, 'modality': 't', 'fairness_model': None, 'epochs':100, 'disc_reg_weight':0.1, 'd_steps': 10})
# quick_start(model = 'VBPR', dataset = 'microlens', config_dict = {'gpu_id': 0, 'modality': 'v', 'fairness_model': None, 'epochs':100, 'disc_reg_weight':0.1, 'd_steps': 10})
# quick_start(model = 'VBPR', dataset = 'microlens', config_dict = {'gpu_id': 1, 'modality': 'a', 'fairness_model': None, 'epochs':100, 'disc_reg_weight':0.1, 'd_steps': 10})


# quick_start(model = 'VBPR', dataset = 'ml1m', config_dict = {'gpu_id': 5, 'modality': 't', 'fairness_model': None, 'epochs':30, 'disc_reg_weight':0.1, 'd_steps': 10})
# quick_start(model = 'VBPR', dataset = 'ml1m', config_dict = {'gpu_id': 6, 'modality': 'v', 'fairness_model': None, 'epochs':30, 'disc_reg_weight':0.1, 'd_steps': 10})
# quick_start(model = 'VBPR', dataset = 'ml1m', config_dict = {'gpu_id': 7, 'modality': 'a', 'fairness_model': None, 'epochs':30, 'disc_reg_weight':0.1, 'd_steps': 10})


