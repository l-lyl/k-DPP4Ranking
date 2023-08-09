import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--weights_path', nargs='?', default='model/',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='beauty',
                        help='Choose a dataset from {anime-1, beauty, CDs}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[16,16,16]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='ngcf',
                        help='Specify the name of model (ngcf).')
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')

    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--node_dropout_flag', type=int, default=1,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[3, 5, 10, 20, 50]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='full',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    
    parser.add_argument('--cate_num', type=int, default=213,
                        help="beauty:213 anime:43 ml:18 cd:340 music:")
    parser.add_argument('--k_size', type=int, default=3,
                        help='k in k-dpp')
    parser.add_argument('--neg_sample_size', type=int, default=3,
                        help='default is equal to k_size') 
    
    parser.add_argument('--is_MF', type=int, default=0, help='use MF or GCN')
    parser.add_argument('--dpp_loss', type=int, default=1, help= '1 ps; 2 nps;')
    parser.add_argument('--seq_sample', type=int, default=1,
                        help='0 randomly sample, 1 select one item and its next ones')
    parser.add_argument('--l_kernel_emb', nargs='?', default='item_kernel.pkl',
                        help='prelearned diverse kernel')

    parser.add_argument('--kernel_sigmoid', type=int, default=0,
                        help='defauld 0: no sig or exp, 1: sigmoid for diverse kernel, 2: exp')
    
    parser.add_argument('--emb_loss', type=int, default=0,
                        help='add normalization loss or not')
    parser.add_argument('--score_exp', type=int, default=1,
                        help='exp for scores')
    parser.add_argument('--dpp_sigma', type=int, default=1,
                        help='sigma for gaussian kernel')
    parser.add_argument('--norm_all_emb', type=int, default=1,
                        help='normalize all embeddings, ps prefers 1, nps prefers 0')
    
    parser.add_argument('--emb_narrow', type=float, default=1,
                        help='narrow down the norm embeddings') #set this to 0.2 and set norm_all_emb=0 if the performance is not good

    return parser.parse_args()
