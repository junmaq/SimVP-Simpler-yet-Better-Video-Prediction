import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor '
                                                                   'computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./cropped_video_frames_dataset')
    parser.add_argument('--train_dir', default='./crack_train')
    parser.add_argument('--val_dir', default='./crack_val')
    parser.add_argument('--dataname', default='crack', choices=['mmnist', 'taxibj', 'crack'])
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[8, 3, 256, 128], type=int,
                        nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj
    parser.add_argument('--hid_S', default=156, type=int)
    parser.add_argument('--hid_T', default=704, type=int)
    parser.add_argument('--N_S', default=6, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=251, type=int)
    parser.add_argument('--log_step', default=4, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)