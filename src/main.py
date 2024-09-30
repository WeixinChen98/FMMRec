# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recommendation_model', '-m', type=str, default='LATTICE', help='name of recommendation models')
    parser.add_argument('--dataset', '-d', type=str, default='ml1m', help='name of datasets')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='gpu_id')
    parser.add_argument('--epochs', '-e', type=int, default=1000, help='epoch')
    parser.add_argument('--fairness_model', '-fm', type=str, default=None, help='name of fairness model')
    parser.add_argument('--d_steps', '-ds', type=int, default=10, help='discriminator update steps')
    parser.add_argument('--disc_epoch', '-de', type=int, default=1000, help='discriminator epoch')
    parser.add_argument('--model_load_path', type=str, default=None, help='the path of the trained model')
    parser.add_argument('--vision_feature_file', '-vf', type=str, default='image_feat.npy', help='vision_feature_file')
    parser.add_argument('--text_feature_file', '-tf', type=str, default='text_feat.npy', help='text_feature_file')
    parser.add_argument('--audio_feature_file', '-af', type=str, default='audio_feat.npy', help='audio_feature_file')

    # FMMR default: compositional
    parser.add_argument('--disc_reg_weight', type=float, default=0.1)
    parser.add_argument('--filter_mode', type=str, default='independent', help='independent or shared filters for user and item representations')
    parser.add_argument('--prompt_mode', type=str, default='add', help='add/concat/none when using shared filters for user and item representations, ignored otherwise')
    parser.add_argument('--weight_i', type = float, default=0.5, help='additional weight for implicit user representation')
    parser.add_argument('--multimodal_information', type=str, default='add', help='add multimodal feature to item embedding before filtering')
    parser.add_argument('--knn_k_uugraph', type=int, default=10)

    config_dict = {
        'gpu_id': 0,
    }

    args = parser.parse_args()

    config_dict.update(vars(args))

    quick_start(recommendation_model=args.recommendation_model, dataset=args.dataset, config_dict=config_dict, save_model=True)



