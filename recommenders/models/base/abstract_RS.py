import numpy as np
import torch
import torch.nn as nn
from data import Data
from evaluator import ProxyEvaluator
from util import DataIterator
from util.cython.tools import float_type, is_ndarray
from util import typeassert, argmax_top_k
from concurrent.futures import ThreadPoolExecutor
from models.base.utils import *
import datetime
import json
import wandb

# define the abstract class for recommender system
class AbstractRS(nn.Module):
    def __init__(self, args, special_args) -> None:
        super(AbstractRS, self).__init__()

        # basic information
        self.args = args
        self.special_args = special_args
        self.device = torch.device(args.cuda)
        self.test_only = args.test_only
        self.candidate = args.candidate

        self.Ks = args.Ks
        self.patience = args.patience
        self.modeltype = args.modeltype
        self.neg_sample = args.neg_sample
        self.inbatch = self.args.infonce == 1 and self.args.neg_sample == -1
    
        # basic hyperparameters
        self.n_layers = args.n_layers
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.epoch
        self.verbose = args.verbose

        # load the data
        self.dataset_name = args.dataset
        try:
            print('from models.'+ args.modeltype + ' import ' + args.modeltype + '_Data')
            exec('from models.'+ args.modeltype + ' import ' + args.modeltype + '_Data') # load special dataset
            self.data = eval(args.modeltype + '_Data(args)') 
        except:
            print("no special dataset")
            self.data = Data(args) # load data from the path
        
        self.n_users = self.data.n_users
        self.n_items = self.data.n_items
        self.train_user_list = self.data.train_user_list
        self.valid_user_list = self.data.valid_user_list
        # = torch.tensor(self.data.population_list).cuda(self.device)
        self.user_pop = torch.tensor(self.data.user_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.item_pop = torch.tensor(self.data.item_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.user_pop_max = self.data.user_pop_max
        self.item_pop_max = self.data.item_pop_max 

        # load the model
        self.running_model = args.modeltype + '_batch' if self.inbatch else args.modeltype
        exec('from models.'+ args.modeltype + ' import ' + self.running_model) # import the model first
        self.model = eval(self.running_model + '(args, self.data)') # initialize the model with the graph
        self.model.cuda(self.device)


        # preparing for saving
        self.preperation_for_saving(args, special_args)
        
        # preparing for evaluation
        # self.not_candidate_dict = self.data.get_not_candidate() # load the not candidate dict
        self.evaluators, self.eval_names = self.get_evaluators(self.data) # load the evaluators


    # the whole pipeline of the training process
    def execute(self):
        
        self.save_args() # save the args
        # write args
        perf_str = str(self.args)
        with open(self.base_path + 'stats.txt','a') as f:
            f.write(perf_str+"\n")

        self.model, self.start_epoch = self.restore_checkpoint(self.model, self.base_path, self.device) # restore the checkpoint

        start_time = time.time()
        # train the model if not test only
        if not self.test_only:
            print("start training") 
            self.train()
            # test the model
            print("start testing")
            self.model = self.restore_best_checkpoint(self.data.best_valid_epoch, self.model, self.base_path, self.device)
        end_time = time.time()
        self.model.eval() # evaluate the best model
        print_str = "The best epoch is % d, total training cost is %.1f" % (max(self.data.best_valid_epoch, self.start_epoch), end_time - start_time)
        with open(self.base_path +'stats.txt', 'a') as f:
            f.write(print_str + "\n")

        n_rets = {}
        for i,evaluator in enumerate(self.evaluators[:]):
            _, __, n_ret = evaluation(self.args, self.data, self.model, self.data.best_valid_epoch, self.base_path, evaluator, self.eval_names[i])
            n_rets[self.eval_names[i]] = n_ret

        self.recommend_top_k()
        self.document_hyper_params_results(self.base_path, n_rets)
        

    def save_args(self):
        # save the args
        with open(self.base_path + '/args.txt', 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

    # define the training process
    def train(self) -> None:
        # TODO
        self.set_optimizer() # get the optimizer
        self.flag = False
        for epoch in range(self.start_epoch, self.max_epoch):
            # print(self.model.embed_user.weight)
            if self.flag: # early stop
                break
            # All models
            t1=time.time()
            losses = self.train_one_epoch(epoch) # train one epoch
            t2=time.time()
            self.document_running_loss(losses, epoch, t2-t1) # report the loss
            if (epoch + 1) % self.verbose == 0: # evaluate the model
                self.eval_and_check_early_stop(epoch)

        visualize_and_save_log(self.base_path +'stats.txt', self.dataset_name)

    #! must be implemented by the subclass
    def train_one_epoch(self, epoch):
        raise NotImplementedError
    
    def preperation_for_saving(self, args, special_args):
        self.formatted_today=datetime.date.today().strftime('%m%d') + '_'

        tn = '1' if args.train_norm else '0'
        pn = '1' if args.pred_norm else '0'
        self.train_pred_mode = 't' + tn + 'p' + pn

        if(self.test_only == False):
            prefix = self.formatted_today + args.saveID
        else:
            prefix = args.saveID
        self.saveID = prefix + '_' + self.train_pred_mode + "_Ks=" + str(args.Ks) + '_patience=' + str(args.patience)\
            + "_n_layers=" + str(args.n_layers) + "_batch_size=" + str(args.batch_size)\
                + "_neg_sample=" + str(args.neg_sample) + "_lr=" + str(args.lr) 
        
        for arg in special_args:
            print(arg, getattr(args, arg))
            self.saveID += "_" + arg + "=" + str(getattr(args, arg))
        
        self.modify_saveID()

        if self.modeltype == 'LightGCN' and self.n_layers == 0:
            self.base_path = './weights/{}/MF/{}'.format(self.dataset_name, self.saveID)
        elif self.n_layers > 0 and self.modeltype != "LightGCN":
            self.base_path = './weights/{}/{}-LGN/{}'.format(self.dataset_name, self.running_model, self.saveID)
        else:
            self.base_path = './weights/{}/{}/{}'.format(self.dataset_name, self.running_model, self.saveID)
        self.checkpoint_buffer=[]
        ensureDir(self.base_path)

    def modify_saveID(self):
        pass

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam([param for param in self.model.parameters() if param.requires_grad == True], lr=self.lr)

    def document_running_loss(self, losses:list, epoch, t_one_epoch, prefix=""):
        loss_str = ', '.join(['%.5f']*len(losses)) % tuple(losses)
        perf_str = prefix + 'Epoch %d [%.1fs]: train==[' % (
                epoch, t_one_epoch) + loss_str + ']'
        with open(self.base_path + 'stats.txt','a') as f:
                f.write(perf_str+"\n")
    
    def document_hyper_params_results(self, base_path, n_rets):
        overall_path = '/'.join(base_path.split('/')[:-1]) + '/'
        hyper_params_results_path = overall_path + self.formatted_today + self.dataset_name + '_' + self.modeltype + '_' + self.args.saveID + '_hyper_params_results.csv'

        results = {'notation': self.formatted_today, 'train_pred_mode':self.train_pred_mode, 'best_epoch': max(self.data.best_valid_epoch, self.start_epoch), 'max_epoch': self.max_epoch, 'Ks': self.Ks, 'n_layers': self.n_layers, 'batch_size': self.batch_size, 'neg_sample': self.neg_sample, 'lr': self.lr}
        for special_arg in self.special_args:
            results[special_arg] = getattr(self.args, special_arg)

        for k, v in n_rets.items():
            if('test_id' not in k):
                for metric in ['recall', 'ndcg', 'hit_ratio']:
                    results[k + '_' + metric] = round(v[metric], 4)
        frame_columns = list(results.keys())
        # load former xlsx
        if os.path.exists(hyper_params_results_path):
            # hyper_params_results = pd.read_excel(hyper_params_results_path)
            hyper_params_results = pd.read_csv(hyper_params_results_path)
        else:
            # Create a new dataframe using the results.
            hyper_params_results = pd.DataFrame(columns=frame_columns)

        hyper_params_results = hyper_params_results._append(results, ignore_index=True)
        # to csv
        hyper_params_results.to_csv(hyper_params_results_path, index=False, float_format='%.4f')
        # hyper_params_results.to_excel(hyper_params_results_path, index=False)

    def recommend_top_k(self):
        test_users = list(self.data.test_user_list.keys())
        if(self.candidate == False):
            dump_dict = merge_user_list([self.data.train_user_list,self.data.valid_user_list])
        recommended_top_k = {}
        recommended_scores = {}
        test_users = DataIterator(test_users, batch_size=self.batch_size, shuffle=False, drop_last=False)
        for batch_id, batch_users in enumerate(test_users):
            if self.data.test_neg_user_list is not None:
                candidate_items = {u:list(self.data.test_user_list[u]) + self.data.test_neg_user_list[u] if u in self.data.test_neg_user_list.keys() else list(self.data.test_user_list[u]) for u in batch_users}

                ranking_score = self.model.predict(batch_users, None)  # (B,N)
                if not is_ndarray(ranking_score, float_type):
                    ranking_score = np.array(ranking_score, dtype=float_type)

                all_items = set(range(ranking_score.shape[1]))
                for idx, user in enumerate(batch_users):
                    # print(max(set(candidate_items[user])), )
                    not_user_candidates = list(all_items - set(candidate_items[user]))
                    ranking_score[idx,not_user_candidates] = -np.inf

                    pos_items = self.data.valid_user_list[user]
                    pos_items = [ x for x in pos_items if not x in self.data.test_user_list[user] ]
                    ranking_score[idx][pos_items] = -np.inf

                    recommended_top_k[user] = argmax_top_k(ranking_score[idx], self.Ks)
                    # ground_truth = self.data.test_user_list[user]
                    # hits = [1 if item in ground_truth else 0 for item in recommended_top_k[user]]
                    # print(sum(hits)/self.Ks)
                    recommended_scores[user] = ranking_score[idx][recommended_top_k[user]]
                    # print('finish one user')
            else:
                ranking_score = self.model.predict(batch_users, None)  # (B,N)
                if not is_ndarray(ranking_score, float_type):
                    ranking_score = np.array(ranking_score, dtype=float_type)
                # set the ranking scores of training items to -inf,
                # then the training items will be sorted at the end of the ranking list.
                
                for idx, user in enumerate(batch_users):
                    dump_items = dump_dict[user]
                    dump_items = [ x for x in dump_items if not x in self.data.test_user_list[user] ]
                    ranking_score[idx][dump_items] = -np.inf

                    recommended_top_k[user] = argmax_top_k(ranking_score[idx], self.Ks)
                    # recommended_scores[user] = ranking_score[idx][recommended_top_k[user]]
                    recommended_scores[user] = ranking_score[idx]
            print('finish recommend one batch', batch_id)

        recommended_top_k = dict(sorted(recommended_top_k.items(), key=lambda x: x[0]))
        # with open(self.base_path + '/recommend_top_k.txt', 'w') as f:
        #     for u, v in recommended_top_k.items():
        #         f.write(str(int(u)))
        #         for i in range(self.Ks):
        #             f.write(' ' + str(int(v[i])) + '+' + str(round(recommended_scores[u][i], 4)))
        #         f.write('\n')
        with open(self.base_path + '/recommend_top_k.txt', 'w') as f:
            for u, v in recommended_top_k.items():
                f.write(str(int(u)))
                for i in range(self.Ks):
                    f.write(' ' + str(int(v[i])))
                f.write('\n')
        print('finish recommend top k')


    
    # define the evaluation process
    def eval_and_check_early_stop(self, epoch):
        self.model.eval()

        for i,evaluator in enumerate(self.evaluators):
            is_best, temp_flag, n_ret = evaluation(self.args, self.data, self.model, epoch, self.base_path, evaluator, self.eval_names[i])
            if(not self.args.no_wandb):
                wandb.log(
                    data = {f"Recall@{self.Ks}": n_ret['recall'], 
                            f"Hit Ratio@{self.Ks}": n_ret['recall'],
                            f"Precision@{self.Ks}": n_ret['precision'],
                            f"NDCG@{self.Ks}": n_ret['ndcg']},
                    step = epoch
                )
            if is_best:
                checkpoint_buffer=save_checkpoint(self.model, epoch, self.base_path, self.checkpoint_buffer, self.args.max2keep)
            
            # early stop?
            if temp_flag:
                self.flag = True
        # checkpoint_buffer=save_checkpoint(self.model, epoch, self.base_path, self.checkpoint_buffer, self.args.max2keep)
        
        self.model.train()
    
    # load the checkpoint
    def restore_checkpoint(self, model, checkpoint_dir, device, force=False, pretrain=False):
        """
        If a checkpoint exists, restores the PyTorch model from the checkpoint.
        Returns the model and the current epoch.
        """
        cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                    if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

        if not cp_files:
            print('No saved model parameters found')
            if force:
                raise Exception("Checkpoint not found")
            else:
                return model, 0,

        epoch_list = []

        regex = re.compile(r'\d+')

        for cp in cp_files:
            epoch_list.append([int(x) for x in regex.findall(cp)][0])

        epoch = max(epoch_list)

    
        if not force:
            print("Which epoch to load from? Choose in range [0, {})."
                .format(epoch), "Enter 0 to train from scratch.")
            print(">> ", end = '')
            # inp_epoch = int(input())

            if self.args.clear_checkpoints:
                print("Clear checkpoint")
                clear_checkpoint(checkpoint_dir)
                return model, 0,

            inp_epoch = epoch
            if inp_epoch not in range(epoch + 1):
                raise Exception("Invalid epoch number")
            if inp_epoch == 0:
                print("Checkpoint not loaded")
                clear_checkpoint(checkpoint_dir)
                return model, 0,
        else:
            print("Which epoch to load from? Choose in range [0, {}).".format(epoch))
            inp_epoch = int(input())
            if inp_epoch not in range(0, epoch):
                raise Exception("Invalid epoch number")

        filename = os.path.join(checkpoint_dir,
                                'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

        print("Loading from checkpoint {}?".format(filename))

        checkpoint = torch.load(filename, map_location = str(device))
        # print("finish load")

        try:
            if pretrain:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint['state_dict'])
            print("=> Successfully restored checkpoint (trained for {} epochs)"
                .format(checkpoint['epoch']))
        except:
            print("=> Checkpoint not successfully restored")
            raise

        return model, inp_epoch
    
    def restore_best_checkpoint(self, epoch, model, checkpoint_dir, device):
        """
        Restore the best performance checkpoint
        """
        cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                    if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

        filename = os.path.join(checkpoint_dir,
                                'epoch={}.checkpoint.pth.tar'.format(epoch))

        print("Loading from checkpoint {}?".format(filename))

        checkpoint = torch.load(filename, map_location = str(device))

        model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)"
            .format(checkpoint['epoch']))

        return model
    
    def get_evaluators(self, data, pop_mask=None):
        #if not self.args.pop_test:
        K_value = self.args.Ks
        if self.args.nodrop: # whether using the enhanced dataset
            eval_train_user_list = data.train_nodrop_user_list
        else:
            eval_train_user_list = data.train_user_list
        
        # if self.args.candidate:
        #     eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[K_value],dump_dict=merge_user_list([data.train_user_list, data.test_user_list]))
        #     eval_test = ProxyEvaluator(data,data.train_user_list,data.test_user_list,top_k=[K_value],dump_dict=merge_user_list([data.train_user_list, data.valid_user_list]), user_neg_test = data.test_neg_user_list)

        # else: 
        #     eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[K_value],dump_dict=merge_user_list([data.train_user_list, data.test_user_list]))  
        #     eval_test = ProxyEvaluator(data,data.train_user_list,data.test_user_list,top_k=[K_value],dump_dict=merge_user_list([data.train_user_list, data.valid_user_list]))


        if self.args.candidate:
            eval_valid = ProxyEvaluator(data,eval_train_user_list,data.valid_user_list,top_k=[K_value],dump_dict=merge_user_list([eval_train_user_list, data.test_user_list]))
            eval_test = ProxyEvaluator(data,eval_train_user_list,data.test_user_list,top_k=[K_value],dump_dict=merge_user_list([eval_train_user_list, data.valid_user_list]), user_neg_test = data.test_neg_user_list)

        else: 
            eval_valid = ProxyEvaluator(data,eval_train_user_list,data.valid_user_list,top_k=[K_value],dump_dict=merge_user_list([eval_train_user_list, data.test_user_list]))  
            eval_test = ProxyEvaluator(data,eval_train_user_list,data.test_user_list,top_k=[K_value],dump_dict=merge_user_list([eval_train_user_list, data.valid_user_list]))

        evaluators=[eval_valid, eval_test]
        eval_names=["valid", "test"]

        return evaluators, eval_names

