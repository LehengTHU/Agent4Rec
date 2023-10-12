import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Overall settings
    parser.add_argument('--simulation_name', type=str, default= 'Test',
                        help='The name of one trial of simulation.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify which gpu to use.')
    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed.')
    parser.add_argument('--items_per_page', type=int, default=4,
                        help='Number of items per page.')
    parser.add_argument('--num_avatars', type=int, default=20,
                        help='Number of avatars for sandbox simulation.')
    parser.add_argument('--execution_mode', type=str, default= 'parallel',
                        choices=['serial', 'parallel'],
                        help='Specify execution mode: serial or parallel.')

    # Only recommend ground truth
    parser.add_argument("--rec_gt", action="store_true",
                        help="whether to recommend ground truth")
    
    # Using wandb
    parser.add_argument("--use_wandb", action="store_true",
                        help="whether to use wandb")
    
    # Only validate the effectiveness of agents
    parser.add_argument("--val_users", action="store_true",
                        help="whether to validate users")
    parser.add_argument('--val_ratio', type=int, default=1,
                        help='Ratio of unobserved items vs ground truth for validation.')
    
    # Advertisement settings
    parser.add_argument("--add_advert", action="store_true",
                        help="whether to add advertisement")
    parser.add_argument("--display_advert", action="store_true",
                        help="whether to display advertisement")
    parser.add_argument('--advert_type', type=str, default='pop_high',
                        choices=['all', 'pop_high', 'pop_low', 'unpop_high', 'unpop_low'],
                        help='Specify advertisement type.')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help='Dataset to use.')

    # Avatar settings
    parser.add_argument('--n_avatars', type=int, default=3,
                        help='How many avatars to simulate.')
    parser.add_argument('--max_pages', type=int, default=1,
                        help='The maximum page number users would like to view')


    # Recommender settings
    parser.add_argument('--model_path', type=str, default= 'Saved',
                        help='Specify model save path.')
    parser.add_argument('--modeltype', type=str, default= 'LightGCN',
                        help='Specify model save path.')

    # others
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument("--pred_norm", action="store_true",
                        help="pred_norm")

    args, _ = parser.parse_known_args()

    return args


