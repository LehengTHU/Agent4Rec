import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                    help='Choose a dataset')
    
    args, _ = parser.parse_known_args()
    return args

args = parse_args()
    
cmds = []
lgn_setting = "--modeltype LightGCN --n_layers 2"
mf_setting = "--modeltype MF --n_layers 0"

cmds.append(f"nohup python train_recommender.py --clear_checkpoints --saveID mf --dataset {args.dataset} {mf_setting} --patience 20 --cuda 0 &> logs/{args.dataset}_origin_mf.log &")
cmds.append(f"nohup python train_recommender.py --clear_checkpoints --saveID lgn --dataset {args.dataset} {lgn_setting} --patience 20 --cuda 1 &> logs/{args.dataset}_origin_lgn.log &")
# cmds.append(f"nohup python train_recommender.py --test_only --no_wandb --modeltype Pop --dataset {args.dataset} &> logs/{args.dataset}_pop.log &")
cmds.append(f"python train_recommender.py --test_only --no_wandb --modeltype Pop --dataset {args.dataset}")

with open("training.sh", "w") as f:
    for cmd in cmds:
        f.write(cmd + "\n")
