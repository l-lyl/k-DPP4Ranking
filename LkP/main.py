
import torch
import torch.optim as optim

from LkP import LkP
from utility.helper import *
from utility.batch_test import *
import pickle as cPickle
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    args.device = torch.device('cuda:' + str(args.gpu_id))

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = LkP(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)
    
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    diverse_emb_file = args.data_path + args.dataset + '/' + args.l_kernel_emb
    lk_param = cPickle.load(open(diverse_emb_file, 'rb'), encoding="latin1")
    lk_tensor = torch.FloatTensor(lk_param['V'])

    lk_emb_i = F.normalize(lk_tensor, p=2, dim=1)
    l_kernel = torch.matmul(lk_emb_i, lk_emb_i.t())
    if args.kernel_sigmoid == 1:
        l_kernel = torch.sigmoid(l_kernel)
    elif args.kernel_sigmoid == 2:
        l_kernel = torch.exp(l_kernel) 

    K = args.k_size
    a_diag = torch.eye(K)*1e-5
    a_diag = a_diag.reshape((1, K, K))
    up_diag = a_diag.repeat(args.batch_size, 1, 1)
        
    loss_loger, rec_loger, ndcg_loger,  cc_loger = [], [], [], []
    for epoch in range(args.epoch):
        if args.dpp_loss != 0:
            batch_kernel = torch.zeros(args.batch_size, args.k_size, args.k_size) #for batch
            batch_nkernel = torch.zeros(args.batch_size, args.k_size, args.k_size)
            batch_lkernel = torch.zeros(args.batch_size, args.k_size + args.neg_sample_size, args.k_size + args.neg_sample_size) #分母  args.k_size*2, args.k_size*2)#
        
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        
        if args.dpp_loss == 1:
            
            for idx in range(n_batch):
                users, pos_items, neg_items = data_generator.sample_dpp(args.k_size, args.seq_sample, args.neg_sample_size)
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                            pos_items,
                                                                            neg_items,
                                                                            drop_flag=args.node_dropout_flag)
                ##construct batch diverse kernels
                batch_pos = np.reshape(pos_items, (args.batch_size, args.k_size))
                batch_neg = np.reshape(neg_items, (args.batch_size, args.neg_sample_size)) #args.k_size
                
                for n in range(args.batch_size):
                    gen_list = batch_pos[n] #targets
                    cond_list = np.concatenate((gen_list, batch_neg[n]), 0)
                    
                    gen_kernel = l_kernel[gen_list][:, gen_list]
                    cond_kernel = l_kernel[cond_list][:, cond_list]
                
                    batch_kernel[n] = gen_kernel    
                    batch_lkernel[n] = cond_kernel
                
                dpp_loss = model.create_kdpp_loss(u_g_embeddings.unsqueeze(1),
                                                 pos_i_g_embeddings,
                                                 neg_i_g_embeddings,
                                                 batch_kernel.to(args.device), 
                                                 batch_lkernel.to(args.device))
                                                 #up_diag, K)
                '''
                dpp_loss = model.create_kdpp_loss_batch(u_g_embeddings.unsqueeze(1),
                                                 pos_i_g_embeddings,
                                                 neg_i_g_embeddings,
                                                 batch_kernel.to(args.device), 
                                                 batch_lkernel.to(args.device),
                                                 up_diag, K)
                '''
                                                 
                optimizer.zero_grad()
                dpp_loss.backward()
                optimizer.step()

                loss += dpp_loss
                
        else:
            for idx in range(n_batch):
                users, pos_items, neg_items = data_generator.sample_dpp(args.k_size, args.seq_sample, args.neg_sample_size)  
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                            pos_items,
                                                                            neg_items,
                                                                            drop_flag=args.node_dropout_flag)
                
                batch_pos = np.reshape(pos_items, (args.batch_size, args.k_size))
                batch_neg = np.reshape(neg_items, (args.batch_size, args.k_size))
                
                for n in range(args.batch_size):
                    gen_list = batch_pos[n] #targets
                    neg_list = batch_neg[n]
                    cond_list = np.concatenate((gen_list, batch_neg[n]), 0)
                    
                    gen_kernel = l_kernel[gen_list][:, gen_list]
                    neg_kernel = l_kernel[neg_list][:, neg_list]
                    cond_kernel = l_kernel[cond_list][:, cond_list]
                
                    batch_kernel[n] = gen_kernel
                    batch_nkernel[n] = neg_kernel
                    batch_lkernel[n] = cond_kernel
                    
                dpp_loss = model.create_kdpp_loss_pn(u_g_embeddings.unsqueeze(1),
                                                 pos_i_g_embeddings, 
                                                 neg_i_g_embeddings,
                                                 batch_kernel.to(args.device), 
                                                 batch_nkernel.to(args.device),
                                                 batch_lkernel.to(args.device))
                optimizer.zero_grad()
                dpp_loss.backward()
                optimizer.step()

                loss += dpp_loss
        
        if epoch < 201:        
            if (epoch + 1) % 40 != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d: train==[%.5f=%.5f + %.5f]' % (
                        epoch, loss, mf_loss, emb_loss)
                    print(perf_str)
                continue
        else:
            if (epoch + 1) % 20 != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d: train==[%.5f=%.5f + %.5f]' % (
                        epoch, loss, mf_loss, emb_loss)
                    print(perf_str)
                continue

        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        ndcg_loger.append(ret['ndcg'])
        cc_loger.append(ret['cc'])

        if args.verbose > 0:
            perf_str = 'Epoch %d: train==[%.5f], recall=[%.5f, %.5f,  %.5f,  %.5f, %.5f], ' \
                       'ndcg=[%.5f, %.5f,  %.5f,  %.5f, %.5f], cc=[%.5f, %.5f,  %.5f,  %.5f, %.5f]' % \
                       (epoch, loss, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['recall'][3], ret['recall'][4],
                        ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][3], ret['ndcg'][-1], ret['cc'][0], ret['cc'][1], ret['cc'][2], ret['cc'][3],ret['cc'][-1])
            print(perf_str)
