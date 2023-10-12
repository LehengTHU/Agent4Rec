import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', nargs='?', default=-1,
                        help='we only want test value.')
    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed.')
    parser.add_argument('--clear_checkpoints', action="store_true",
                        help='Whether clear the earlier checkpoints.')
    parser.add_argument("--candidate", action="store_true",
                        help="whether using the candidate set")
    parser.add_argument('--test_only', action="store_true",
                        help='Whether to test only.')
    parser.add_argument('--data_path', nargs='?', default='../datasets/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--regs', type=float, default=1e-5,
                        help='Regularization.')
    parser.add_argument('--epoch', type=int, default=2000,
                        help='Number of epoch.')
    parser.add_argument('--Ks', type = int, default= 20,
                        help='Evaluate on Ks optimal items.')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--saveID', type=str, default="",
                        help='Specify model save path.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping point.')
    parser.add_argument('--checkpoint', type=str, default='./',
                        help='Specify model save path.')
    parser.add_argument('--modeltype', type=str, default= 'MF',
                        help='Specify model save path.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify which gpu to use.')
    parser.add_argument('--IPStype', type=str, default='cn',
                        help='Specify the mode of weighting')
    parser.add_argument('--n_layers', type=int, default=0,
                        help='Number of GCN layers')
    parser.add_argument('--max2keep', type=int, default=1,
                        help='max checkpoints to keep')
    parser.add_argument('--infonce', type=int, default=0,
                        help='whether to use infonce loss or not')
    parser.add_argument('--neg_sample',type=int,default=1)
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers in data loader')
    parser.add_argument("--train_norm", action="store_true",
                        help="train_norm")
    parser.add_argument("--pred_norm", action="store_true",
                        help="pred_norm")
    
    parser.add_argument("--nodrop", action="store_true",
                        help="whether to drop out the enhanced training dataset")
    parser.add_argument("--no_wandb", action="store_true",
                        help="whether to use wandb")

    args, _ = parser.parse_known_args()

    # INFONCE
    if(args.modeltype == 'InfoNCE'):
        parser.add_argument('--tau', type=float, default=0.1,
                        help='temperature parameter')
    
    # MultVAE
    if(args.modeltype == 'MultVAE'):
        parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='total anneal steps')
        parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='anneal cap')
        parser.add_argument('--p_dim0', type=int, default=200,
                        help='p_dim0')
        parser.add_argument('--p_dim1', type=int, default=600,
                        help='p_dim1')
    
    args_full, _ = parser.parse_known_args()
    special_args = list(set(vars(args_full).keys()) - set(vars(args).keys()))
    special_args.sort()

    return args_full, special_args


