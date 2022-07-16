from .dataloader_taxibj import load_data as load_taxibj
from .dataloader_moving_mnist import load_data as load_mmnist
from .dataloader_crack import load_data as load_crack


def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, train_dir, val_dir, **kwargs):
    if dataname == 'taxibj':
        return load_taxibj(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'crack':
        return load_crack(batch_size, val_batch_size,  num_workers, data_root, train_dir, val_dir)
