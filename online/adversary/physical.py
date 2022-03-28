import argparse
import os.path as osp
import os
import pickle
import json
from datetime import datetime
import sys

import numpy as np

from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

sys.path.append("/home/ubuntu/Project/Meta_GAN_gr")
from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
from knockoff.victim.blackbox import Blackbox
import knockoff.config as cfg
from knockoff.adversary.adaptive import AdaptiveAdversary
from knockoff.adversary.jacobian import JacobianAdversary

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


ROOT = "/home/ubuntu/Project/Meta_GAN_gr"

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# TODO: specify the return type
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path: str):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class Dataset():

    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.loader = default_loader
        self.transform = transform
        self.batch_size = 64

        self.transferset = []  # List of tuples [(img_path, output_probs)]
        self.idx_set = set()

        self._restart()

    def _restart(self):
        np.random.seed(DEFAULT_SEED)
        torch.manual_seed(DEFAULT_SEED)
        torch.cuda.manual_seed(DEFAULT_SEED)

        self.idx_set = set(range(len(self.paths)))
        self.transferset = []

    def __call__(self, index):
        path = self.paths[index]
        sample = self.loader(path)
        vector = torch.from_numpy(self.labels[index])

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, vector

    def __len__(self):
        return len(self.paths)

    def get_transfer(self, budget):
        start_B = 0
        end_B = budget
        with tqdm(total=budget) as pbar:
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):
                idxs = np.random.choice(list(self.idx_set), replace=False,
                                        size=min(self.batch_size, budget - len(self.transferset)))
                self.idx_set = self.idx_set - set(idxs)

                if len(self.idx_set) == 0:
                    print('=> Query set exhausted. Now repeating input examples.')
                    self.idx_set = set(range(len(self.paths)))

                for i in idxs:
                    img_t_i = self.paths[i]
                    self.transferset.append((img_t_i, torch.from_numpy(self.labels[i]).cpu().squeeze()))

                pbar.update(len(idxs))

        return self.transferset


def assemble_transferset_gtsrb(vectors, transform, budget):
    # queryset
    test_dir = "/home/ubuntu/Project/Meta_GAN_gr/GTSRB"
    tbp_paths = os.listdir(test_dir)
    get_key = lambda i: int(i.split('.')[0])
    tbp_paths = sorted(tbp_paths, key=get_key)
    tbp_paths = [osp.join(test_dir, path_x) for path_x in tbp_paths]
    gtsrb_dataset = Dataset(tbp_paths, vectors, transform)
    transferset = gtsrb_dataset.get_transfer(budget)

    return transferset


device_id = 1

if device_id >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# ----------- Set up queryset --> Load images
valid_datasets = datasets.__dict__.keys()
queryset_name = 'GTSRB'
if queryset_name not in valid_datasets:
    raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
modelfamily = datasets.dataset_to_modelfamily[queryset_name]
transform = datasets.modelfamily_to_transforms[modelfamily]['test']


# --------- Load the output vectors obtained through the commercial api
out_path = os.path.join(ROOT, queryset_name)
if not os.path.exists(out_path): os.mkdir(out_path)
transfer_out_path = osp.join(out_path, 'transferset.pickle')
api_output_path = '/home/ubuntu/Project/Meta_GAN_gr/complete_test.npy'
api_output_vectors = np.load(api_output_path)
# in the form of [[], ..., []]
transferset = assemble_transferset_gtsrb(api_output_vectors, transform, budget=len(api_output_vectors))

with open(transfer_out_path, 'wb') as wf:
    pickle.dump(transferset, wf)


class RandomAdversary(object):
    def __init__(self, blackbox, queryset, batch_size=8):
        self.blackbox = blackbox
        self.queryset = queryset

        self.n_queryset = len(self.queryset)
        self.batch_size = batch_size
        self.idx_set = set()

        self.transferset = []  # List of tuples [(img_path, output_probs)]

        self._restart()

    def _restart(self):
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        self.idx_set = set(range(len(self.queryset)))
        self.transferset = []

    def get_transferset(self, budget):
        start_B = 0
        end_B = budget
        with tqdm(total=budget) as pbar:
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):
                idxs = np.random.choice(list(self.idx_set), replace=False,
                                        size=min(self.batch_size, budget - len(self.transferset)))
                self.idx_set = self.idx_set - set(idxs)

                if len(self.idx_set) == 0:
                    print('=> Query set exhausted. Now repeating input examples.')
                    self.idx_set = set(range(len(self.queryset)))

                x_t = torch.stack([self.queryset[i][0] for i in idxs]).to(self.blackbox.device)
                y_t = self.blackbox(x_t).cpu()

                if hasattr(self.queryset, 'samples'):
                    # Any DatasetFolder (or subclass) has this attribute
                    # Saving image paths are space-efficient
                    img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                else:
                    # Otherwise, store the image itself
                    # But, we need to store the non-transformed version
                    img_t = [self.queryset.data[i] for i in idxs]
                    if isinstance(self.queryset.data[0], torch.Tensor):
                        img_t = [x.numpy() for x in img_t]

                for i in range(x_t.size(0)):
                    img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
                    self.transferset.append((img_t_i, y_t[i].cpu().squeeze()))

                pbar.update(x_t.size(0))

        return self.transferset


def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('policy', metavar='PI', type=str, help='Policy to use while training', default='random',
                        choices=['random', 'adaptive'])
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--budget', metavar='N', type=int, help='Size of transfer set to construct',
                        required=True)
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X))', required=True, default='ImageFolder')
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=64)
    parser.add_argument('--root', metavar='DIR', type=str, help='Root directory for ImageFolder', default=None)
    parser.add_argument('--modelfamily', metavar='TYPE', type=str, help='Model family', default=None)
    # parser.add_argument('--topk', metavar='N', type=int, help='Use posteriors only from topk classes',
    #                     default=None)
    # parser.add_argument('--rounding', metavar='N', type=int, help='Round posteriors to these many decimals',
    #                     default=None)
    # parser.add_argument('--tau_data', metavar='N', type=float, help='Frac. of data to sample from Adv data',
    #                     default=1.0)
    # parser.add_argument('--tau_classes', metavar='N', type=float, help='Frac. of classes to sample from Adv data',
    #                     default=1.0)
    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    knockoff_utils.create_dir(out_path)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up queryset
    queryset_name = params['queryset']
    valid_datasets = datasets.__dict__.keys()
    if queryset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[queryset_name] if params['modelfamily'] is None else params['modelfamily']
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    if queryset_name == 'ImageFolder':
        assert params['root'] is not None, 'argument "--root ROOT" required for ImageFolder'
        queryset = datasets.__dict__[queryset_name](root=params['root'], transform=transform)
    else:
        queryset = datasets.__dict__[queryset_name](train=True, transform=transform)

    # ----------- Initialize blackbox
    # TO DO: use api as blackbox
    blackbox_dir = params['victim_model_dir']
    blackbox = Blackbox.from_modeldir(blackbox_dir, device)

    # ----------- Initialize adversary
    batch_size = params['batch_size']
    nworkers = params['nworkers']
    transfer_out_path = osp.join(out_path, 'transferset.pickle')
    if params['policy'] == 'random':
        adversary = RandomAdversary(blackbox, queryset, batch_size=batch_size)
    elif params['policy'] == 'adaptive':
        adversary = AdaptiveAdversary(blackbox, queryset, batch_size=batch_size)
    elif params['policy'] == 'jacobian':
        adversary = JacobianAdversary(blackbox, queryset, batch_size=batch_size)
    else:
        raise ValueError("Unrecognized policy")

    print('=> constructing transfer set...')
    transferset = adversary.get_transferset(params['budget'])
    with open(transfer_out_path, 'wb') as wf:
        pickle.dump(transferset, wf)
    print('=> transfer set ({} samples) written to: {}'.format(len(transferset), transfer_out_path))

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params_transfer.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()