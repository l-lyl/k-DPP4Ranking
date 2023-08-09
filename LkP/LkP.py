
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class LkP(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(LkP, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size
        
        self.dpp_loss = args.dpp_loss
        self.k_size = args.k_size
        self.emb_loss = args.emb_loss
        self.score_exp = args.score_exp

        self.norm_adj = norm_adj

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]
        
        self.emb_narrow = args.emb_narrow
        self.norm_all_emb = args.norm_all_emb
        
        self.neg_size = args.neg_sample_size
        
        self.is_MF = args.is_MF
        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        # xavier init
        
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))
        })
        '''
        initializer = nn.init.normal_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size), mean=0.0, std=0.01)),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size), mean=0.0, std=0.01))
        })
        '''
        
        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))
    
    def log_elementary_symmetric_polynomial(self, evals, K):
        N = len(evals)
        e = torch.zeros(K+1)
        e[1:] = -np.inf
        log_evals = torch.log(evals)
        for n in range(1, N + 1):
            enew = torch.zeros(K+1)
            enew[1:] = -np.inf
            enew[1:] = torch.logaddexp(e[1:], log_evals[n-1] + e[:-1])
            e = enew
        return e[-1]
    
    def elementary_symmetric_polynomial(self, evals, K):
        N = len(evals)
        e = torch.zeros(K+1)
        e[0] = 1
        for n in range(1, N + 1):
            enew = torch.zeros(K + 1)
            enew[0] = 1
            enew[1:] = e[1:] + evals[n-1] * e[:-1]
            e = enew
        return e[-1]

    def create_kdpp_loss(self, user_embs, pos_embs, neg_embs, kernel, lkernel):

        pos_scores = torch.sum(user_embs * pos_embs, dim=-1) 
        neg_scores = torch.sum(user_embs * neg_embs, dim=-1) 
        
        dpp_lhs = []
        size = user_embs.shape[0]
        gset_scores = torch.cat((pos_scores, neg_scores), 1)  

        for n in range(size):
            if self.score_exp == 1:
                pos_q = torch.diag_embed(torch.exp(pos_scores[n]))  
                set_q = torch.diag_embed(torch.exp(gset_scores[n]))
            else:
                pos_q = torch.diag_embed(pos_scores[n])  
                set_q = torch.diag_embed(gset_scores[n])
            
            pos_k = torch.mm(torch.mm(pos_q, kernel[n]), pos_q) 
            set_k = torch.mm(torch.mm(set_q, lkernel[n]), set_q).cpu()
            
            pos_det = torch.det(pos_k.cpu() + torch.eye(pos_k.shape[0])*1e-6).to(self.device)
           
            evals, evecs = torch.eig(set_k, eigenvectors=True) ##use batch format of pytorch if the version is new
            real_evals = evals[:, 0]

            denominator = self.elementary_symmetric_polynomial(real_evals, pos_k.shape[0])
            logp = torch.log(pos_det/denominator.to(self.device)) 
            
            dpp_lhs.append(logp)
            
        if self.emb_loss == 1:
            regularizer = (torch.norm(user_embs) ** 2
                    + torch.norm(pos_embs) ** 2
                    + torch.norm(neg_embs) ** 2) / 2
            emb_loss = self.decay * regularizer / self.batch_size
            return -torch.mean(torch.stack(dpp_lhs)) + emb_loss
        
        return -torch.mean(torch.stack(dpp_lhs))
    
    ##mini_batch format
    def create_kdpp_loss_batch(self, user_embs, pos_embs, neg_embs, kernel, lkernel, up_diag, K):

        pos_scores = torch.sum(user_embs * pos_embs, dim=-1)
        neg_scores = torch.sum(user_embs * neg_embs, dim=-1)
        gset_scores = torch.cat((pos_scores, neg_scores), 1)
        
        pos_q = torch.diag_embed(torch.exp(pos_scores))  
        set_q = torch.diag_embed(torch.exp(gset_scores))
        
        l_up = torch.bmm(torch.bmm(pos_q, kernel), pos_q)
        l_down = torch.bmm(torch.bmm(set_q, lkernel), set_q) 
        
        pos_det = torch.det(l_up.cpu() + up_diag).to(self.device)
        
        dpp_lhs = []
        size = user_embs.shape[0]
        
        for n in range(size):
            
            evals, evecs = torch.eig(l_down[n].cpu(), eigenvectors=True) ##change this to batch format 
            real_evals = evals[:, 0]
            
            denominator = self.elementary_symmetric_polynomial(real_evals, K)
            logp = torch.log(pos_det[n]/denominator.to(self.device))
            
            dpp_lhs.append(logp)
            
        return -torch.mean(torch.stack(dpp_lhs))
    
    def create_kdpp_loss_pn(self, user_embs, pos_embs, neg_embs, kernel, nkernel, lkernel):

        pos_scores = torch.sum(user_embs * pos_embs, dim=-1)
        neg_scores = torch.sum(user_embs * neg_embs, dim=-1)
        
        dpp_lhs = []
        size = user_embs.shape[0]
        gset_scores = torch.cat((pos_scores, neg_scores), 1)
        for n in range(size):
            if self.score_exp == 1:
                pos_q = torch.diag_embed(torch.exp(pos_scores[n]))  
                neg_q = torch.diag_embed(torch.exp(neg_scores[n]))  
                set_q = torch.diag_embed(torch.exp(gset_scores[n]))
            else:
                pos_q = torch.diag_embed(pos_scores[n])  
                neg_q = torch.diag_embed(neg_scores[n])  
                set_q = torch.diag_embed(gset_scores[n])
           
            diag = torch.eye(set_q.shape[0])*1e-6  
            pos_k = torch.mm(torch.mm(pos_q, kernel[n]), pos_q)
            neg_k = torch.mm(torch.mm(neg_q, nkernel[n]), neg_q)
            set_k = torch.mm(torch.mm(set_q, lkernel[n]), set_q) + diag.to(self.device)
            
            pos_det = torch.det(pos_k.cpu() + torch.eye(pos_k.shape[0])*1e-6).to(self.device) 
            neg_det = torch.det(neg_k.cpu() + torch.eye(neg_k.shape[0])*1e-6).to(self.device)
            
            evals, evecs = torch.eig(set_k.cpu(), eigenvectors=True) ##change this to batch format  
            real_evals = evals[:, 0]
            
            denominator = self.elementary_symmetric_polynomial(real_evals, pos_k.shape[0])
            logp = torch.log(pos_det/denominator.to(self.device)) + torch.log(1 - neg_det/denominator.to(self.device))  
                
            dpp_lhs.append(logp)
            
        return -torch.mean(torch.stack(dpp_lhs))

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, pos_items, neg_items, drop_flag=True):
        if self.is_MF == 1:
            u_g_embeddings = F.normalize(self.embedding_dict['user_emb'][users, :], p=2, dim=1)
            pos_i_g_embeddings = F.normalize(self.embedding_dict['item_emb'][pos_items, :], p=2, dim=1)
            neg_i_g_embeddings = F.normalize(self.embedding_dict['item_emb'][neg_items, :], p=2, dim=1)
            
            if (self.dpp_loss != 0 and self.dpp_loss != 2) and (u_g_embeddings.shape[0] == self.batch_size):
                pos_i_g_embeddings = pos_i_g_embeddings.reshape(u_g_embeddings.shape[0], self.k_size, u_g_embeddings.shape[1])
                neg_i_g_embeddings = neg_i_g_embeddings.reshape(u_g_embeddings.shape[0], self.neg_size, u_g_embeddings.shape[1]) #self.k_size,

            return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]

            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]

            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
            
            if self.norm_all_emb == 0: 
                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings*self.emb_narrow]  ##try emb_narrow to avoid the scores being big 
            else:
                all_embeddings += [ego_embeddings]
        
        if self.norm_all_emb == 0:
            all_embeddings = torch.cat(all_embeddings, 1)
            u_g_embeddings = all_embeddings[:self.n_user, :]
            i_g_embeddings = all_embeddings[self.n_user:, :]
        else:
            all_embeddings = torch.cat(all_embeddings, 1)
            norm_all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            u_g_embeddings = norm_all_embeddings[:self.n_user, :]
            i_g_embeddings = norm_all_embeddings[self.n_user:, :]

        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        if (self.dpp_loss != 0 and self.dpp_loss != 2) and (u_g_embeddings.shape[0] == self.batch_size):
            pos_i_g_embeddings = pos_i_g_embeddings.reshape(u_g_embeddings.shape[0], self.k_size, u_g_embeddings.shape[1])
            neg_i_g_embeddings = neg_i_g_embeddings.reshape(u_g_embeddings.shape[0], self.neg_size, u_g_embeddings.shape[1]) #self.k_size,

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
