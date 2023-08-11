
# k-DPP4Ranking

A PyTorch implementation of LkP

# Requirements
* Python 2 or 3
* [PyTorch v0.4+](https://github.com/pytorch/pytorch)
* Numpy
* SciPy

# Usage
1. Install required packages.
2. use 3_item_dpp_emb.py to create the pre-learned diverse kernel
3. run <code>python main.py --dataset beauty --regs [1e-5] --embed_size 16 --layer_size [16,16,16] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --dpp_loss 1 --k_size 3 --cate_num 213 --gpu_id 1 --score_exp 1 --seq_sample 1 --pre_kernel 1 --dpp_sigma 1 --emb_narrow 1 --norm_all_emb 1</code>

# Configurations


#### Data


#### Model Args 


# Citation


