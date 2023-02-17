import torch
import random
import numpy as np
import os
import typesentry

"""
Parts of this code was based upon the work from this repository:
https://github.com/JiaweiSheng/FAAN
Modifications have been made.
"""
tc1 = typesentry.Config()
typed = tc1.typed  # decorator to check function arguments at runtime
is_typed = tc1.is_type  # equivalent of isinstance()


def seed_everything(seed=2040):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
