
import numpy as np
import os
import torch
import torch.nn as nn
import scipy as sp

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from common.init import xavier_normal_initialization
import torch.nn.functional as F
from time import gmtime, strftime
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood


class BFMMR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(BFMMR, self).__init__(config, dataloader)

        self.base_model = config['recommendation_model']
        self.neg_slope = config['neg_slope']
        self.num_features = len(config['feature_columns'])
        self.feature_columns = config['feature_columns']

        self.filter_mode = config['filter_mode']
        self.prompt_mode = config['prompt_mode']


        # load model path
        self.model_path = "{}{}_{}_{}_{}_{}_vision={}_text={}_audio={}_discWeight={}_".format(config['model_path'], config['dataset'], config['recommendation_model'], config['fairness_model'], config['filter_mode'], config['prompt_mode'], config['vision_feature_file'], config['text_feature_file'], config['audio_feature_file'], config['disc_reg_weight'])
        for i in config['hyper_parameters']:
            self.model_path += '_{}={}'.format(i, config[i])
        current_time = strftime("%Y-%m-%d_%H_%M_%S", gmtime())
        self.model_path += current_time + '.pt'

        self.logger.info(self.model_path)

        pretrained_user_representation = np.load(os.path.join(self.dataset_path, self.base_model + '_user_representation.npy'), allow_pickle=True)
        self.user_representation = torch.FloatTensor(pretrained_user_representation).to(self.device)
        # self.user_representation.requires_grad = True 
        pretrained_item_representation = np.load(os.path.join(self.dataset_path, self.base_model + '_item_representation.npy'), allow_pickle=True)
        self.item_representation = torch.FloatTensor(pretrained_item_representation).to(self.device)
        # self.item_representation.requires_grad = True 


        self.interaction_matrix = dataloader.inter_matrix(form='coo').astype(np.float32)
        self.interaction_matrix = self.sparse_mx_to_torch_sparse_tensor(self.interaction_matrix).float().to_dense()
        row_sums = self.interaction_matrix.sum(axis=-1)
        self.interaction_matrix = self.interaction_matrix / row_sums[:, np.newaxis]
        self.interaction_matrix = self.interaction_matrix.to(self.device)


        # 0 indicates user and 1 indicate user-centric item representation
        # switch embedding
        self.useritem_prompt_embedding = nn.Embedding(2, config['user_representation_size'])

        if self.filter_mode == 'shared' and self.prompt_mode == 'concat':
            self._init_sensitive_filter(filter_num = self.num_features, embedding_dim = config['user_representation_size'] * 2, out_dim = config['user_representation_size'])
        else:
            self._init_sensitive_filter(filter_num = self.num_features, embedding_dim = config['user_representation_size'], out_dim = config['user_representation_size'])

        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton


        self.knn_k = config['knn_k_uugraph']
        self.n_mg_uugraph_layers = config['n_mg_uugraph_layers']
        self.mg_weight = config['mg_weight']
        self.mg_uugraph_text_weight = config['mg_uugraph_text_weight']
        self.mg_uugraph_image_weight = config['mg_uugraph_image_weight']
        self.mg_uugraph_audio_weight = config['mg_uugraph_audio_weight']

        self.dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        biased_image_adj_file = os.path.join(self.dataset_path, 'biased_image_adj_{}.pt'.format(self.knn_k))
        biased_text_adj_file = os.path.join(self.dataset_path, 'biased_text_adj_{}.pt'.format(self.knn_k))
        biased_audio_adj_file = os.path.join(self.dataset_path, 'biased_audio_adj_{}.pt'.format(self.knn_k))

        filtered_image_adj_file = os.path.join(self.dataset_path, 'filtered_image_adj_{}.pt'.format(self.knn_k))
        filtered_text_adj_file = os.path.join(self.dataset_path, 'filtered_text_adj_{}.pt'.format(self.knn_k))
        filtered_audio_adj_file = os.path.join(self.dataset_path, 'filtered_audio_adj_{}.pt'.format(self.knn_k))


        # load biased features here
        self.biased_v_feat, self.biased_t_feat, self.biased_a_feat = None, None, None
        if config['biased_vision_feature_file'] is not None:
            biased_v_feat_file_path = os.path.join(self.dataset_path, config['biased_vision_feature_file'])
            if os.path.isfile(biased_v_feat_file_path):
                self.biased_v_feat = torch.from_numpy(np.load(biased_v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)
            
            if os.path.exists(biased_image_adj_file):
                biased_image_adj = torch.load(biased_image_adj_file)
            else:
                user_biased_v_feat = torch.mm(self.interaction_matrix, self.biased_v_feat)
                biased_image_adj = build_sim(user_biased_v_feat)
                biased_image_adj = build_knn_neighbourhood(biased_image_adj, topk=self.knn_k)
                biased_image_adj = compute_normalized_laplacian(biased_image_adj)
                torch.save(biased_image_adj, biased_image_adj_file)
            self.biased_image_original_adj = biased_image_adj.cuda()

        if  config['biased_text_feature_file'] is not None:
            biased_t_feat_file_path = os.path.join(self.dataset_path, config['biased_text_feature_file'])
            if os.path.isfile(biased_t_feat_file_path):
                self.biased_t_feat = torch.from_numpy(np.load(biased_t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)

            if os.path.exists(biased_text_adj_file):
                biased_text_adj = torch.load(biased_text_adj_file)
            else:
                user_biased_t_feat = torch.mm(self.interaction_matrix, self.biased_t_feat)
                biased_text_adj = build_sim(user_biased_t_feat)
                biased_text_adj = build_knn_neighbourhood(biased_text_adj, topk=self.knn_k)
                biased_text_adj = compute_normalized_laplacian(biased_text_adj)
                torch.save(biased_text_adj, biased_text_adj_file)
            self.biased_text_original_adj = biased_text_adj.cuda()

        if  config['biased_audio_feature_file'] is not None:    
            biased_a_feat_file_path = os.path.join(self.dataset_path, config['biased_audio_feature_file'])
            if os.path.isfile(biased_a_feat_file_path):
                self.biased_a_feat = torch.from_numpy(np.load(biased_a_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)

            if os.path.exists(biased_audio_adj_file):
                biased_audio_adj = torch.load(biased_audio_adj_file)
            else:
                user_biased_t_feat = torch.mm(self.interaction_matrix, self.biased_t_feat)
                biased_audio_adj = build_sim(user_biased_t_feat)
                biased_audio_adj = build_knn_neighbourhood(biased_audio_adj, topk=self.knn_k)
                biased_audio_adj = compute_normalized_laplacian(biased_audio_adj)
                torch.save(biased_audio_adj, biased_audio_adj_file)
            self.biased_audio_original_adj = biased_audio_adj.cuda()

        assert self.biased_v_feat is not None or self.biased_t_feat is not None or  self.biased_a_feat is not None, 'Biased features all NONE'

        # load filtered features here
        self.filtered_v_feat, self.filtered_t_feat, self.filtered_a_feat = None, None, None
        if config['filtered_vision_feature_file'] is not None:
            filtered_v_feat_file_path = os.path.join(self.dataset_path, config['filtered_vision_feature_file'])
            if os.path.isfile(filtered_v_feat_file_path):
                self.filtered_v_feat = torch.from_numpy(np.load(filtered_v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)
            
            if os.path.exists(filtered_image_adj_file):
                filtered_image_adj = torch.load(filtered_image_adj_file)
            else:
                user_filtered_v_feat = torch.mm(self.interaction_matrix, self.filtered_v_feat)
                filtered_image_adj = build_sim(user_filtered_v_feat)
                filtered_image_adj = build_knn_neighbourhood(filtered_image_adj, topk=self.knn_k)
                filtered_image_adj = compute_normalized_laplacian(filtered_image_adj)
                torch.save(filtered_image_adj, filtered_image_adj_file)
            self.filtered_image_original_adj = filtered_image_adj.cuda()

        if  config['filtered_text_feature_file'] is not None:
            filtered_t_feat_file_path = os.path.join(self.dataset_path, config['filtered_text_feature_file'])
            if os.path.isfile(filtered_t_feat_file_path):
                self.filtered_t_feat = torch.from_numpy(np.load(filtered_t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)

            if os.path.exists(filtered_text_adj_file):
                filtered_text_adj = torch.load(filtered_text_adj_file)
            else:
                user_filtered_t_feat = torch.mm(self.interaction_matrix, self.filtered_t_feat)
                filtered_text_adj = build_sim(user_filtered_t_feat)
                filtered_text_adj = build_knn_neighbourhood(filtered_text_adj, topk=self.knn_k)
                filtered_text_adj = compute_normalized_laplacian(filtered_text_adj)
                torch.save(filtered_text_adj, filtered_text_adj_file)
            self.filtered_text_original_adj = filtered_text_adj.cuda()

        if  config['filtered_audio_feature_file'] is not None:    
            filtered_a_feat_file_path = os.path.join(self.dataset_path, config['filtered_audio_feature_file'])
            if os.path.isfile(filtered_a_feat_file_path):
                self.filtered_a_feat = torch.from_numpy(np.load(filtered_a_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)

            if os.path.exists(filtered_audio_adj_file):
                filtered_audio_adj = torch.load(filtered_audio_adj_file)
            else:
                user_filtered_t_feat = torch.mm(self.interaction_matrix, self.filtered_t_feat)
                filtered_audio_adj = build_sim(user_filtered_t_feat)
                filtered_audio_adj = build_knn_neighbourhood(filtered_audio_adj, topk=self.knn_k)
                filtered_audio_adj = compute_normalized_laplacian(filtered_audio_adj)
                torch.save(filtered_audio_adj, filtered_audio_adj_file)
            self.filtered_audio_original_adj = filtered_audio_adj.cuda()


    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        batch_users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        fair_user_representation, fair_item_representation = self.get_fair_representation()


        if self.base_model == 'LATTICE':
            u_g_embeddings = fair_user_representation[batch_users]
            pos_i_g_embeddings = fair_item_representation[pos_items]
            neg_i_g_embeddings = fair_item_representation[neg_items]

            batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                        neg_i_g_embeddings)
            return batch_mf_loss + batch_emb_loss + batch_reg_loss
        elif self.base_model == 'DRAGON':
            user_tensor = fair_user_representation[batch_users]
            pos_item_tensor = fair_item_representation[pos_items, :]
            neg_item_tensor = fair_item_representation[neg_items, :]
            pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
            neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
            loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
            return loss_value
            
    def full_sort_predict(self, interaction):
        user = interaction[0]
        fair_user_representation, fair_item_representation = self.get_fair_representation()
        user_e = fair_user_representation[user]
        all_item_e = fair_item_representation
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score
     

    def save_pretrained_representation(self):
        fair_user_representation, fair_item_representation = self.get_fair_representation()
        fair_user_representation = fair_user_representation.cpu().detach().numpy()
        fair_item_representation = fair_item_representation.cpu().detach().numpy()
        with open(os.path.join(self.dataset_path, self.base_model + '_BFMMR_user_representation.npy'), 'wb') as fu:
            np.save(fu, fair_user_representation, allow_pickle=True)
        with open(os.path.join(self.dataset_path, self.base_model + '_BFMMR_item_representation.npy'), 'wb') as fi:
            np.save(fi, fair_item_representation, allow_pickle=True)


    def get_user_and_graph_embedding(self, users):
        fair_user_representation, fair_item_representation = self.get_fair_representation()
        fair_item_representation = torch.mm(self.interaction_matrix[users], fair_item_representation)
        fair_user_representation = fair_user_representation[users]
        return fair_user_representation, fair_item_representation

    # for disc evaluation
    def get_user_embedding(self, users):
        return self.fair_user_representation[users]

    # for disc evaluation
    def get_user_aggregated_item_embedding(self, users):
        fair_item_representation = self.fair_item_representation
        fair_item_representation = torch.mm(self.interaction_matrix[users], fair_item_representation)
        return fair_item_representation


    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def _init_sensitive_filter(self, filter_num, embedding_dim = 64, out_dim = 64):
        def get_sensitive_filter(embed_dim, out_dim):
            sequential = nn.Sequential(
                nn.Linear(embed_dim, out_dim),
                nn.LeakyReLU(self.neg_slope, inplace=True),
                nn.Linear(out_dim, out_dim),
            )
            return sequential
            
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        if self.filter_mode == 'independent':
            self.user_filters = nn.ModuleDict({str(f): get_sensitive_filter(embedding_dim, out_dim) for f in self.feature_columns})
            for _, f in self.user_filters.items():
                f.apply(init_weights)

            self.item_filters = nn.ModuleDict({str(f): get_sensitive_filter(embedding_dim, out_dim) for f in self.feature_columns})
            for _, f in self.item_filters.items():
                f.apply(init_weights)

        elif self.filter_mode == 'shared':
            self.filter_dict = nn.ModuleDict({str(f): get_sensitive_filter(embedding_dim, out_dim) for f in self.feature_columns})
            for _, f in self.filter_dict.items():
                    f.apply(init_weights)
        
        elif self.filter_mode == 'user-only':
            self.user_filters = nn.ModuleDict({str(f): get_sensitive_filter(embedding_dim, out_dim) for f in self.feature_columns})
            for _, f in self.user_filters.items():
                f.apply(init_weights)

    # LATTICE
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    # SLMRec
    def infonce(self, users_emb, pos_emb):
        users_emb = torch.nn.functional.normalize(users_emb, dim=1)
        pos_emb = torch.nn.functional.normalize(pos_emb, dim=1)
        logits = torch.mm(users_emb, pos_emb.T)
        logits /= self.temp
        labels = torch.tensor(list(range(users_emb.shape[0]))).to(self.device)
        return self.infonce_criterion(logits, labels)

    def get_fair_representation(self):
        fair_user_representation = None
        fair_item_representation = None
        u_embedding = self.user_representation
        i_embedding = self.item_representation

        biased_adj = self.mg_uugraph_text_weight * self.biased_text_original_adj + self.mg_uugraph_image_weight * self.biased_image_original_adj + self.mg_uugraph_audio_weight * self.biased_audio_original_adj
        biased_h = self.user_representation
        for i in range(self.n_mg_uugraph_layers):
            biased_h = torch.mm(biased_adj, biased_h)

        filtered_adj = self.mg_uugraph_text_weight * self.filtered_text_original_adj + self.mg_uugraph_image_weight * self.filtered_image_original_adj + self.mg_uugraph_audio_weight * self.filtered_audio_original_adj
        filtered_h = self.user_representation
        for i in range(self.n_mg_uugraph_layers):
            filtered_h = torch.mm(filtered_adj, filtered_h)


        # u_embedding = u_embedding - self.mg_weight * F.normalize(biased_h, p=2, dim=1)
        u_embedding = u_embedding + self.mg_weight * (filtered_h - biased_h)

        if self.filter_mode == 'independent':
            for _, filter in self.user_filters.items():
                if fair_user_representation is None:
                    fair_user_representation = filter(u_embedding)
                else:
                    fair_user_representation += filter(u_embedding)

            fair_user_representation /= float(self.num_features)

            for _, filter in self.item_filters.items():
                if fair_item_representation is None:
                    fair_item_representation = filter(i_embedding)
                else:
                    fair_item_representation += filter(i_embedding)

            fair_item_representation /= float(self.num_features)

        elif self.filter_mode == 'shared':
            if self.prompt_mode == 'none':
                pass
            elif self.prompt_mode == 'add':
                user_prompt = torch.ones(self.n_users, dtype=torch.long).to(self.device)
                user_prompt = self.useritem_prompt_embedding(user_prompt)
                u_embedding = u_embedding + user_prompt

                item_prompt = torch.zeros(self.n_items, dtype=torch.long).to(self.device)
                item_prompt = self.useritem_prompt_embedding(item_prompt)
                i_embedding = i_embedding + item_prompt
            elif self.prompt_mode == 'concat':
                user_prompt = torch.ones(self.n_users, dtype=torch.long).to(self.device)
                user_prompt = self.useritem_prompt_embedding(user_prompt)
                u_embedding = torch.concat([user_prompt, u_embedding], dim=-1)

                item_prompt = torch.zeros(self.n_items, dtype=torch.long).to(self.device)
                item_prompt = self.useritem_prompt_embedding(item_prompt)
                i_embedding = torch.concat([item_prompt, i_embedding], dim=-1)
            else:
                raise ValueError("prompt mode is not in [none, add, concat]")

            for _, filter in self.filter_dict.items():
                if fair_user_representation is None:
                    fair_user_representation = filter(u_embedding)
                else:
                    fair_user_representation += filter(u_embedding)

            fair_user_representation /= float(self.num_features)

            for _, filter in self.filter_dict.items():
                if fair_item_representation is None:
                    fair_item_representation = filter(i_embedding)
                else:
                    fair_item_representation += filter(i_embedding)

            fair_item_representation /= float(self.num_features)
        
        elif self.filter_mode == 'user-only':
            for _, filter in self.user_filters.items():
                if fair_user_representation is None:
                    fair_user_representation = filter(u_embedding)
                else:
                    fair_user_representation += filter(u_embedding)

            fair_user_representation /= float(self.num_features)
            fair_item_representation = i_embedding
        else:
            raise ValueError("filter mode is not in [independent, shared, user-only]")

        self.fair_user_representation = fair_user_representation
        self.fair_item_representation = fair_item_representation
        return fair_user_representation, fair_item_representation