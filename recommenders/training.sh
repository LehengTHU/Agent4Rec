nohup python train_recommender.py --clear_checkpoints --saveID lgn --dataset ml-1m_real_0 --modeltype MF --n_layers 0 --patience 20 --cuda 0 &> logs/ml-1m_real_0_origin_mf.log &
nohup python train_recommender.py --clear_checkpoints --saveID lgn --dataset ml-1m_real_0 --modeltype LightGCN --n_layers 2 --patience 20 --cuda 1 &> logs/ml-1m_real_0_origin_lgn.log &
nohup python train_recommender.py --test_only --no_wandb --modeltype Pop --dataset ml-1m_real_0 &> logs/ml-1m_real_0_pop.log &
