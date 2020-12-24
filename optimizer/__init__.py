from .LookAhead import Lookahead
from .RAdam import RAdam
from .Ranger import Ranger
from torch import optim


def get_optimizer(method, module):
    if method == 'adam':
        optimizer = optim.Adam(module.parameters(), lr=1e-3, betas=(0.95, 0.999))
    elif method == 'sgd':  # 从头训练 lr=0.1 fine_tune lr=0.01
        # optimizer = optim.SGD(module.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005*24)  # Yolo
        optimizer = optim.SGD(module.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0001)  # FRCNN
    elif method == 'radam':
        optimizer = RAdam(module.parameters(), lr=1e-3, betas=(0.95, 0.999))
    elif method == 'lookahead':
        optimizer = Lookahead(module.parameters())
    elif method == 'ranger':
        optimizer = Ranger(module.parameters(), lr=1e-3)
    else:
        raise NotImplementedError

    return optimizer