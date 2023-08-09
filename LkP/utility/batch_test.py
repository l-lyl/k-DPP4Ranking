
import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import heapq

args = parse_args() 
Ks = eval(args.Ks)

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
CATE_MAP = data_generator.iidcate_map
BATCH_SIZE = args.batch_size
CATE_NUM = args.cate_num

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    cate_set = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
        cates = CATE_MAP[i]
        cate_set.append(cates)
    return r, cate_set

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    cate_set = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
        cates = CATE_MAP[i]
        cate_set.append(cates)
    return r, cate_set

def get_performance(user_pos_test, r,  Ks, cate_set):
    recall, ndcg, cc = [], [], []

    for K in Ks:
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K, user_pos_test))
        cc.append(metrics.cc_at_k(cate_set, K, CATE_NUM))

    return {'recall': np.array(recall),
            'ndcg': np.array(ndcg), 'cc':np.array(cc)}

def test_one_user(rating, u):

    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, cate_set = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, cate_set = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, Ks, cate_set)


def test(model, users_to_test, drop_flag=False, batch_test_flag=False):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'cc': np.zeros(len(Ks))}
    
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)

    count = 0

    if batch_test_flag:
        # batch-item test
        n_item_batchs = ITEM_NUM // i_batch_size + 1
        rate_batch = np.zeros(shape=(len(test_users), ITEM_NUM))

        i_count = 0
        for i_batch_id in range(n_item_batchs):
            i_start = i_batch_id * i_batch_size
            i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

            item_batch = range(i_start, i_end)

            if drop_flag == False:
                u_g_embeddings, pos_i_g_embeddings, _ = model(test_users,
                                                                item_batch,
                                                                [],
                                                                drop_flag=False)
                i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
            else:
                u_g_embeddings, pos_i_g_embeddings, _ = model(test_users,
                                                                item_batch,
                                                                [],
                                                                drop_flag=True)
                i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

            rate_batch[:, i_start: i_end] = i_rate_batch
            i_count += i_rate_batch.shape[1]

        assert i_count == ITEM_NUM

    else:
        # all-item test
        ### removed the pool, batch_users, and test on all users and items
        item_batch = range(ITEM_NUM)

        if drop_flag == False:
            u_g_embeddings, pos_i_g_embeddings, _ = model(test_users,
                                                            item_batch,
                                                            [],
                                                            drop_flag=False)
            rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
        else:
            u_g_embeddings, pos_i_g_embeddings, _ = model(test_users,
                                                            item_batch,
                                                            [],
                                                            drop_flag=True)
            rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
    
    batch_result = []
    for n in range(len(test_users)):
        batch_result.append(test_one_user(rate_batch.numpy()[n], test_users[n]))
    count += len(batch_result)

    for re in batch_result:
        result['recall'] += re['recall']/n_test_users
        result['ndcg'] += re['ndcg']/n_test_users
        result['cc'] += re['cc']/n_test_users

    assert count == n_test_users
    return result
