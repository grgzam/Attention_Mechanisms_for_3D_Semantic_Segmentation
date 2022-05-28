import numpy as np
#import torchpack.distributed as dist

__all__ = ['cosine_schedule_with_warmup']


def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size):
    batch_size *= 1

    warmup_iters = 0

    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
        ratio = (k - warmup_iters) / (num_epochs * iter_per_epoch)
#        print( 0.5 * (1 + np.cos(np.pi * ratio)))
        return 0.5 * (1 + np.cos(np.pi * ratio))

