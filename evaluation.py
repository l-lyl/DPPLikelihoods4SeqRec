import numpy as np
import math

def _compute_apk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not list(targets):
        return 0.0

    return score / min(len(targets), k)

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def cc_at_k(cc, k, CATE_NUM):
    cates = set()
    for i in range(k):
        if i > (len(cc)-1):
            break
        for c in cc[i]:
           cates.add(c)
    return len(cates) / CATE_NUM

def _compute_precision_recall(targets, predictions, k, iidcate_map, cate_num):

    pred = predictions[:k]
    r = []
    cc = []
    for i in pred:
        if i in targets:
            r.append(1)
        else:
            r.append(0)
        if i == 0:
            continue
        else:
            cc.append(iidcate_map[i-1])

    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    ndcg = ndcg_at_k(r, k) 
    cc = cc_at_k(cc, k, cate_num)
    return precision, recall, ndcg, cc

def cond_greedy_dpp(kernel_matrix, cond_set, topn, epsilon=1E-10):
    item_size = kernel_matrix.shape[0]
    max_length = len(cond_set) + topn
    cis = np.zeros((max_length, item_size))  
    di2s = np.copy(np.diag(kernel_matrix))
    k = 0
    for iid in cond_set:
        ci_optimal = cis[:k, iid]
        di_optimal = math.sqrt(di2s[iid])
        elements = kernel_matrix[iid, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        k += 1
    selected_items = cond_set
    selected_item = iid
    di2s[cond_set[:-1]] = -np.inf
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items[-topn:]

def greedy_dpp(kernel_matrix, max_length, epsilon=1E-12):
    
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items

def evaluate_ranking(model, test, config, l_kernel, cate, train=None, k=10):
    """
    Compute Precision@k, Recall@k scores and average precision (AP).
    One score is given for every user with interactions in the test
    set, representing the AP, Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, rated items in
        interactions will be excluded.
    k: int or array of int,
        The maximum number of predicted items
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if not isinstance(k, list):
        ks = [k]
    else:
        ks = k

    precisions = [list() for _ in range(len(ks))]
    recalls = [list() for _ in range(len(ks))]
    ndcgs = [list() for _ in range(len(ks))]
    ccs = [list() for _ in range(len(ks))]
    apks = list()

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue
        
        predictions = -model.predict(user_id)
        if train is not None:
                rated = set(train[user_id].indices)
        else:
            rated = []
                
        if config.dpp_generation == 0:
            predictions = predictions.argsort()
            predictions = [p for p in predictions if p not in rated]
        ## use dpp to generate items by MAP inference for evaluation
        else:
            predictions = -predictions[1:]
           
            kernel = l_kernel.cpu().numpy()
            sig_rating = 1/(1+np.exp(-predictions))
            prob = sig_rating / np.sum(sig_rating)
            
            u_kernel = np.dot(np.dot(np.diag(prob), kernel), np.diag(prob))
            if 0 in rated:
                rated.remove(0)
            re_rated = [i-1 for i in list(rated)]
            predictions = cond_greedy_dpp(u_kernel, re_rated, max(ks)) 
            predictions = np.array(predictions) + 1
            
        targets = row.indices  
        if 0 in targets:
            print('there is 0')

        for i, _k in enumerate(ks):
            precision, recall, ndcg, cc = _compute_precision_recall(targets, predictions, _k, cate, config.cate_num)
            precisions[i].append(precision)
            recalls[i].append(recall)
            ndcgs[i].append(ndcg)
            ccs[i].append(cc)

        apks.append(_compute_apk(targets, predictions, k=np.inf))

    precisions = [np.array(i) for i in precisions]
    recalls = [np.array(i) for i in recalls]

    if not isinstance(k, list):
        precisions = precisions[0]
        recalls = recalls[0]

    return precisions, recalls, ndcgs, ccs
