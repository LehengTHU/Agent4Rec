import numpy as np
from tqdm import tqdm
import os
from parse import parse_args
from simulation.utils import fix_seeds

from simulation.avatar import Avatar
from simulation.arena import Arena
import wandb


# load model
if __name__ == '__main__':
    args = parse_args()
    # print(args)
    fix_seeds(args.seed) # set random seed

    if(args.use_wandb):
        wandb.init(
            # set the wandb project where this run will be logged
            project = "sandbox",
            name = args.simulation_name,
            group = args.dataset
        )

    arena_ = Arena(args)
    arena_.excute()

    print('Simulation finished!')
    if(args.use_wandb):
        wandb.finish()

