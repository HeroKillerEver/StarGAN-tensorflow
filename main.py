from data_loader import DataLoader
import random
import numpy as np
from solver import Solver
import argparse
import os

parser = argparse.ArgumentParser(description='A tensroflow implementation for Star-GAN', epilog='#' * 75)

########## Model configuration ##########
parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset). Default: 5')
parser.add_argument('--crop_size', type=int, default=178, help='crop size for images. Default: 178')
parser.add_argument('--image_size', type=int, default=128, help='image resolution. Default: 128')
parser.add_argument('--g_conv_num', type=int, default=64, help='number of conv filters in the first layer of G. Default: 64')
parser.add_argument('--d_conv_num', type=int, default=64, help='number of conv filters in the first layer of D. Default: 64')
parser.add_argument('--g_res_num', type=int, default=6, help='number of residual blocks in G. Default: 6')
parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss. Default: 1')
parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss. Default: 10')
parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty. Default: 10')


########## Training configuration ##########
parser.add_argument('--gpus', default='0', type=str, help='gpu to use: 0, 1, 2, 3, 4 or 0,1,2. Default: 0')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate. Default: 1e-4')
parser.add_argument('--batch', default=16, type=int, help='batch size. Default: 16')
parser.add_argument('--epochs', default=50, type=int, help='num of epochs. Default: 50')
parser.add_argument('--seed', default=2018, type=int, help='random seed number. Default: 2018')
parser.add_argument('--mode', type=str, default='train', help='mode, train or test. Default: train')
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
parser.add_argument('--resume', action='store_true', help='restore from the latest checkpoint')
parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')

########## Directories configuration ##########
parser.add_argument('--image_dir', type=str, default='./data/CelebA_nocrop/images')
parser.add_argument('--attr_dir', type=str, default='./data/')
parser.add_argument('--model_save_dir', type=str, default='checkpoints', help='directories to save models, Default: checkpoints')
parser.add_argument('--model_name', type=str, default='stargan', help='checkpoint name, Default: stargan')
parser.add_argument('--result_dir', type=str, default='results', help='directories to save results, Default: results')
parser.add_argument('--log_dir', default='log', type=str, help='directories to save logs, Default: log')


config = parser.parse_args()


def main():

    if not os.path.exists(config.model_save_dir):
        os.mkdir(config.model_save_dir)

    if not os.path.exists(config.result_dir):
        os.mkdir(config.result_dir)

    random.seed(config.seed)
    print 'Loading and Creating data generator...'
    data_gen = DataLoader(config.image_dir, config.attr_dir, config.selected_attrs, config.batch, config.mode)
    print 'Done...!'
    solver = Solver(data_gen, config)

    if config.mode == 'train':
        solver.train()
    else:
        solver.test()

if __name__ == '__main__':
    main()