import random
import os
import torch
import numpy as np


def seed_everything(seed=42, deterministic=True):
    ''' 
    seed everything (os, np, torch and torch.cuda). 
    deterministic = True : using deterministic algo to make exp reproducible
    deterministic = False: using cudnn.benchmark to speed up training
    Example:
        >>> seed_everything(42) 
    '''
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True