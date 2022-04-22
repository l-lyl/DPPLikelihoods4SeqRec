import argparse
from time import time

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import pickle as cPickle

from caser import Caser
from evaluation import evaluate_ranking
from interactions import Interactions
from utils import *


class Recommender(object):
    """
    Contains attributes and methods that needed to train a sequential
    recommendation model. Models are trained by many tuples of
    (users, sequences, targets, negatives) and negatives are from negative
    sampling: for any known tuple of (user, sequence, targets), one or more
    items are randomly sampled to act as negatives.


    Parameters
    ----------

    n_iter: int,
        Number of iterations to run.
    batch_size: int,
        Minibatch size.
    l2: float,
        L2 loss penalty, also known as the 'lambda' of l2 regularization.
    neg_samples: int,
        Number of negative samples to generate for each targets.
        If targets=3 and neg_samples=3, then it will sample 9 negatives.
    learning_rate: float,
        Initial learning rate.
    use_cuda: boolean,
        Run the model on a GPU or CPU.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self,
                 n_iter=None,
                 batch_size=None,
                 l2=None,
                 neg_samples=None,
                 learning_rate=None,
                 use_cuda=False,
                 model_args=None):

        # model related
        self._num_items = None
        self._num_users = None
        self._net = None
        self.model_args = model_args

        # learning related
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples
        self._device = torch.device("cuda" if use_cuda else "cpu")

        # rank evaluation related
        self.test_sequence = None
        self._candidate = dict()

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences

        self._net = Caser(self._num_users,
                          self._num_items,
                          self.model_args).to(self._device)

        self._optimizer = optim.Adam(self._net.parameters(),
                                     weight_decay=self._l2,
                                     lr=self._learning_rate)

    def fit(self, train, test, cate, config, verbose=False):
        """
        The general training loop to fit the model

        Parameters
        ----------

        train: :class:`spotlight.interactions.Interactions`
            training instances, also contains test sequences
        test: :class:`spotlight.interactions.Interactions`
            only contains targets for test sequences
        verbose: bool, optional
            print the logs
        """
        ##################################
        # read pre-learned kernel
        ###################################
        lk_param = cPickle.load(open(config.l_kernel_emb, 'rb'), encoding="latin1")
        lk_tensor = torch.FloatTensor(lk_param['V']).to(self._device)
        
        lk_emb_i = F.normalize(lk_tensor, p=2, dim=1)
        l_kernel = torch.matmul(lk_emb_i, lk_emb_i.t())
        
        #l_kernel = torch.sigmoid(l_kernel)  #this line is optional; use this line, if encounter non-invertible or nan problem
        
        #l_kernel_un = torch.matmul(lk_tensor, lk_tensor.t()) ##un-normalized pre-learned kernel

        # convert to sequences, targets and users
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)

        L, T = train.sequences.L, train.sequences.T

        n_train = sequences_np.shape[0]

        output_str = 'total training instances: %d' % n_train
        print(output_str)

        if not self._initialized:
            self._initialize(train)

        start_epoch = 0
        pre_list = []
        for epoch_num in range(start_epoch, self._n_iter):

            t1 = time()

            # set model to training mode
            self._net.train()

            users_np, sequences_np, targets_np = shuffle(users_np,
                                                         sequences_np,
                                                         targets_np)

            negatives_np = self._generate_negative_samples(users_np, train, n=self._neg_samples)

            # convert numpy arrays to PyTorch tensors and move it to the corresponding devices
            users, sequences, targets, negatives = (torch.from_numpy(users_np).long(),
                                                    torch.from_numpy(sequences_np).long(),
                                                    torch.from_numpy(targets_np).long(),
                                                    torch.from_numpy(negatives_np).long())

            users, sequences, targets, negatives = (users.to(self._device),
                                                    sequences.to(self._device),
                                                    targets.to(self._device),
                                                    negatives.to(self._device))

            epoch_loss = 0.0

            for (minibatch_num,
                 (batch_users,
                  batch_sequences,
                  batch_targets,
                  batch_negatives)) in enumerate(minibatch(users,
                                                           sequences,
                                                           targets,
                                                           negatives,
                                                           batch_size=self._batch_size)):
                items_to_predict = torch.cat((batch_targets, batch_negatives, batch_sequences), 1)
                items_prediction = self._net(batch_sequences,
                                             batch_users,
                                             items_to_predict)
                
                (targets_prediction, negatives_prediction,
                 seq_prediction) = torch.split(items_prediction,
                                                      [batch_targets.size(1),
                                                      batch_negatives.size(1),
                                                      batch_sequences.size(1)], dim=1)

                self._optimizer.zero_grad()
                
                if config.dpp_loss == 0:
                    # compute the binary cross-entropy loss
                    positive_loss = -torch.mean(
                        torch.log(torch.sigmoid(targets_prediction)))
                    negative_loss = -torch.mean(
                        torch.log(1 - torch.sigmoid(negatives_prediction)))
                    loss = positive_loss + negative_loss

                ###############################################
                # compute the dpp set loss
                ###############################################
                # DSL
                elif config.dpp_loss == 1:
                    dpp_lhs = []
                    size = targets_prediction.shape[0]
                    batch_sets = torch.cat((batch_targets, batch_negatives), 1)
                    batch_predictions = torch.cat((targets_prediction, negatives_prediction), 1)
                    #minibatch format
                    if config.batch_format == 1:
                        batch_pos_kernel = torch.zeros(size, config.T, config.T).cuda()
                        batch_set_kernel = torch.zeros(size, config.T + config.neg_samples, config.T + config.neg_samples).cuda()
                        
                        for n in range(size):
                            batch_pos_kernel[n] = l_kernel[batch_targets[n]-1][:, batch_targets[n]-1]
                            batch_set_kernel[n] = l_kernel[batch_sets[n]-1][:, batch_sets[n]-1]
                        
                        batch_pos_q = torch.diag_embed(torch.exp(targets_prediction))  #can also try sigmoid in some cases
                        batch_set_q = torch.diag_embed(torch.exp(batch_predictions))
                        
                        batch_pos_kernel = torch.bmm(torch.bmm(batch_pos_q, batch_pos_kernel), batch_pos_q)
                        batch_set_kernel = torch.bmm(torch.bmm(batch_set_q, batch_set_kernel), batch_set_q)
                        
                        p_diag = torch.eye(config.T)*1e-5
                        pa_diag = p_diag.reshape((1, config.T, config.T))
                        pbatch_diag = pa_diag.repeat(size, 1, 1)
                        
                        s_diag = torch.eye(config.T+config.neg_samples)
                        sa_diag = s_diag.reshape((1, config.T + config.neg_samples, config.T + config.neg_samples))
                        sbatch_diag = sa_diag.repeat(size, 1, 1)
                        
                        batch_pos_det = torch.det(batch_pos_kernel.cpu() + pbatch_diag).cuda()
                        batch_set_det = torch.det(batch_set_kernel.cpu() + sbatch_diag).cuda()
                        
                        dpp_loss = torch.log(batch_pos_det/batch_set_det)
                        loss = -torch.mean(dpp_loss)
                    else:
                        for n in range(size):
                            pos_q = torch.diag_embed(torch.exp(targets_prediction[n]))  
                            set_q = torch.diag_embed(torch.exp(batch_predictions[n]))
                            
                            pos_l_kernel = l_kernel[batch_targets[n]-1][:, batch_targets[n]-1]
                            set_l_kernel = l_kernel[batch_sets[n]-1][:, batch_sets[n]-1]
                            
                            pos_k = torch.mm(torch.mm(pos_q, pos_l_kernel), pos_q)
                            set_k = torch.mm(torch.mm(set_q, set_l_kernel), set_q)
                
                            pos_det = torch.det(pos_k.cpu() + torch.eye(len(batch_targets[n]))*1e-5).cuda() 
                            set_det = torch.det(set_k.cpu() + torch.eye(len(batch_sets[n]))).cuda()

                            dpp_loss = torch.log(pos_det/set_det)
                            
                            dpp_lhs.append(dpp_loss)
                        loss = -torch.mean(torch.stack(dpp_lhs))
                # CDSL
                elif config.dpp_loss == 2:
                    dpp_lhs = []
                    size = targets_prediction.shape[0]
                    set_items = torch.cat((batch_sequences, batch_targets, batch_negatives), 1)
                    set_predictions = torch.cat((seq_prediction, targets_prediction, negatives_prediction), 1)
                    
                    pos_items = torch.cat((batch_sequences, batch_targets), 1)
                    pos_predictions = torch.cat((seq_prediction, targets_prediction), 1) #L+T
                    if config.batch_format == 1:
                        batch_pos_kernel = torch.zeros(size, config.L + config.T, config.L + config.T).cuda()
                        batch_set_kernel = torch.zeros(size, config.L + config.T + config.neg_samples, config.L + config.T + config.neg_samples).cuda()
                        
                        for n in range(size):
                            batch_pos_kernel[n] = l_kernel[pos_items[n]-1][:, pos_items[n]-1]
                            batch_set_kernel[n] = l_kernel[set_items[n]-1][:, set_items[n]-1]
                        
                        batch_pos_q = torch.diag_embed(torch.exp(pos_predictions))
                        batch_set_q = torch.diag_embed(torch.exp(set_predictions))
                        
                        batch_pos_kernel = torch.bmm(torch.bmm(batch_pos_q, batch_pos_kernel), batch_pos_q)
                        batch_set_kernel = torch.bmm(torch.bmm(batch_set_q, batch_set_kernel), batch_set_q)
                        
                        p_diag = torch.eye(config.L + config.T)*1e-3
                        pa_diag = p_diag.reshape((1, config.L + config.T, config.L + config.T))
                        pbatch_diag = pa_diag.repeat(size, 1, 1)
                        
                        s_diag = torch.diag_embed(torch.FloatTensor([1e-3]*config.L+[1]*(config.neg_samples+config.T)))
                        sa_diag = s_diag.reshape((1, config.L + config.T + config.neg_samples, config.L + config.T + config.neg_samples))
                        sbatch_diag = sa_diag.repeat(size, 1, 1)
                        
                        batch_pos_det = torch.det(batch_pos_kernel.cpu() + pbatch_diag).cuda()
                        batch_set_det = torch.det(batch_set_kernel.cpu() + sbatch_diag).cuda()
                        
                        dpp_loss = torch.log(batch_pos_det/batch_set_det)
                        loss = -torch.mean(dpp_loss)
                    else:
                        diag_I = torch.diag_embed(torch.FloatTensor([1e-3]*config.L+[1]*(config.neg_samples+config.T)))
                        diag_posI = torch.diag_embed(torch.FloatTensor([1e-3]*(config.L+config.T)))
                        for n in range(size):
                            pos_q = torch.diag_embed(torch.exp(pos_predictions[n]))  
                            set_q = torch.diag_embed(torch.exp(set_predictions[n]))

                            pos_l_kernel = l_kernel[pos_items[n]-1][:, pos_items[n]-1]
                            set_l_kernel = l_kernel[set_items[n]-1][:, set_items[n]-1]
                            
                            pos_k = torch.mm(torch.mm(pos_q, pos_l_kernel), pos_q)
                            set_k = torch.mm(torch.mm(set_q, set_l_kernel), set_q)
                
                            pos_det = torch.det(pos_k.cpu() + diag_posI).cuda() 
                            set_det = torch.det(set_k.cpu() + diag_I).cuda()

                            dpp_loss = torch.log(pos_det/set_det)
                            dpp_lhs.append(dpp_loss)
                        loss = -torch.mean(torch.stack(dpp_lhs))
                
                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1

            t2 = time()
            if verbose:
                if (epoch_num+1) % 10 == 0:
                    precision, recall, ndcg, cc = evaluate_ranking(self, test, config, l_kernel, cate, train, k=[3, 5, 10])
                    output_str = "Epoch %d [%.1f s], loss=%.4f, " \
                                "prec@3=%.4f, *prec@5=%.4f, prec@10=%.4f, " \
                                "recall@3=%.4f, recall@5=%.4f, recall@10=%.4f, " \
                                "ndcg@3=%.4f, ndcg@5=%.4f, ndcg@10=%.4f, " \
                                "*cc@3=%.4f, cc@5=%.4f, cc@10=%.4f, [%.1f s]" % (epoch_num + 1,
                                                                                            t2 - t1,
                                                                                            epoch_loss,
                                                                                            np.mean(precision[0]),
                                                                                            np.mean(precision[1]),
                                                                                            np.mean(precision[2]),
                                                                                            np.mean(recall[0]),
                                                                                            np.mean(recall[1]),
                                                                                            np.mean(recall[2]),
                                                                                            np.mean(ndcg[0]),
                                                                                            np.mean(ndcg[1]),
                                                                                            np.mean(ndcg[2]),
                                                                                            np.mean(cc[0]),
                                                                                            np.mean(cc[1]),
                                                                                            np.mean(cc[2]),
                                                                                            time() - t2)
                    
                    print(output_str)
            else:
                output_str = "Epoch %d [%.1f s]\tloss=%.4f [%.1f s]" % (epoch_num + 1,
                                                                        t2 - t1,
                                                                        epoch_loss,
                                                                        time() - t2)
                print(output_str)
        
    def _generate_negative_samples(self, users, interactions, n):
        
        """
        Sample negative from a candidate set of each user. The
        candidate set of each user is defined by:
        {All Items} \ {Items Rated by User}

        Parameters
        ----------

        users: array of np.int64
            sequence users
        interactions: :class:`spotlight.interactions.Interactions`
            training instances, used for generate candidates
        n: int
            total number of negatives to sample for each sequence
        """

        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not self._candidate:
            all_items = np.arange(interactions.num_items - 1) + 1  # 0 for padding
            train = interactions.tocsr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))

        for i, u in enumerate(users_):
            for j in range(n):
                x = self._candidate[u]
                negative_samples[i, j] = x[
                    np.random.randint(len(x))]

        return negative_samples

    def predict(self, user_id, item_ids=None):
        """
        Make predictions for evaluation: given a user id, it will
        first retrieve the test sequence associated with that user
        and compute the recommendation scores for items.

        Parameters
        ----------

        user_id: int
           users id for which prediction scores needed.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.
        """

        if self.test_sequence is None:
            raise ValueError('Missing test sequences, cannot make predictions')

        # set model to evaluation model
        self._net.eval()
        with torch.no_grad():
            sequences_np = self.test_sequence.sequences[user_id, :]
            sequences_np = np.atleast_2d(sequences_np)

            if item_ids is None:
                item_ids = np.arange(self._num_items).reshape(-1, 1)

            sequences = torch.from_numpy(sequences_np).long()
            item_ids = torch.from_numpy(item_ids).long()
            user_id = torch.from_numpy(np.array([[user_id]])).long()

            user, sequences, items = (user_id.to(self._device),
                                      sequences.to(self._device),
                                      item_ids.to(self._device))

            out = self._net(sequences,
                            user,
                            items,
                            for_pred=True)

        return out.cpu().numpy().flatten()
    
    def sigma(self, x):
        res = 1 - torch.exp(-model_config.sigma_alpha*x)
        return res

def get_cates_map(cate_file):
    iidcate_map = {}  #iid:cates
    ## movie_id:cate_ids, cate_ids is not only one
    with open(cate_file) as f_cate:
        for l in f_cate.readlines():
            if len(l) == 0: break
            l = l.strip('\n')
            items = [int(i) for i in l.split(' ')]
            iid, cate_ids = items[0], items[1:]
            iidcate_map[iid] = cate_ids
    return iidcate_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_root', type=str, default='datasets/beauty/train_3.txt')
    parser.add_argument('--test_root', type=str, default='datasets/beauty/test_3.txt')
    parser.add_argument('--cateid_root', type=str, default='datasets/beauty/cate.txt')
    parser.add_argument('--l_kernel_emb', type=str, default='datasets/beauty/item_kernel_3.pkl')
    parser.add_argument('--cate_num', type=int, default=213)
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3, help="consistent with the postfix of dataset")
    # dpp arguments
    parser.add_argument('--neg_samples', type=int, default=3, help="Z")
    parser.add_argument('--dpp_loss', type=int, default=2, help="0:cross-entropy, 1:DSL, 2:CDSL")
    parser.add_argument('--batch_format', type=int, default=1, help="use minibatch format for dpp loss or not")
    parser.add_argument('--dpp_generation', type=int, default=0, help="whether use dpp MAP inference to generate items for evaluation")
    # train arguments
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.001, help="[0.0005 0.001 0.0015], default 0.001") 
    parser.add_argument('--l2', type=float, default=1e-4)  
    parser.add_argument('--use_cuda', type=str2bool, default=True)

    config = parser.parse_args()

    # model dependent arguments
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=50)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')
    model_parser.add_argument('--sigma_alpha', type=float, default=0.01)

    model_config = model_parser.parse_args()
    model_config.L = config.L

    # set seed
    set_seed(config.seed,
             cuda=config.use_cuda)

    # load dataset
    train = Interactions(config.train_root)
    # transform triplets to sequence representation
    train.to_sequence(config.L, config.T)

    test = Interactions(config.test_root,
                        user_map=train.user_map,
                        item_map=train.item_map)

    cate = get_cates_map(config.cateid_root)

    print(config)
    print(model_config)
    # fit model
    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        l2=config.l2,
                        neg_samples=config.neg_samples,
                        model_args=model_config,
                        use_cuda=config.use_cuda)

    model.fit(train, test, cate, config, verbose=True)
