import sys, os
from torch.nn import Module
from typing import List, Tuple, Set
import pickle
from subset_selection_strategy import RandomSelectionStrategy, KCenterGreedyApproach
import online.utils.model as model_utils
from online.adversary.train import get_optimizer
from online.models import zoo
import argparse
from typing import Dict, Any, List, Tuple, Union
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import sys, os
from online import datasets
import online.utils as knockoff_utils
import online.config as cfg
from online.victim import *
from online.victim.blackbox import Blackbox
from online.models import zoo
from torch import Tensor

BBOX_CHOICES = ['none', 'topk', 'rounding',
                'reverse_sigmoid', 'reverse_sigmoid_wb',
                'rand_noise', 'rand_noise_wb',
                'mad', 'mad_wb','edm', 'am']


def parser_dealer(option: Dict[str, bool]) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    if option['transfer']:
        parser.add_argument('policy', metavar='PI', type=str, help='Policy to use while training',
                            choices=['random', 'adaptive'])
        parser.add_argument('--budget', metavar='N', type=int, help='Size of transfer set to construct',
                            required=True)
        parser.add_argument('--out_dir', metavar='PATH', type=str,
                            help='Destination directory to store transfer set', required=True)
        parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X))', required=True)

    if option['active']:
        parser.add_argument('strategy', metavar='S', type=str, help='Active Sample Strategy',
                            choices=['kcenter', 'random', 'dfal'])
        parser.add_argument('--metric', metavar="M", type=str, help='K-Center method distance metric',
                            choices=['euclidean', 'manhattan', 'l1', 'l2'], default='euclidean')
        parser.add_argument('--initial-size', metavar='N', type=int, help='Active Learning Initial Sample Size',
                            default=100)
        parser.add_argument('--budget-per-iter', metavar='N', type=int, help='budget for every iteration',
                            default=100)
        parser.add_argument('--iterations', metavar='N', type=int, help='iteration times',
                            default=10)
    if option['defense']:
        parser.add_argument('defense', metavar='TYPE', type=str, help='Type of defense to use',
                            choices=BBOX_CHOICES, default='none')
        parser.add_argument('defense_args', metavar='STR', type=str,
                            help='Blackbox arguments in format "k1:v1,k2:v2,..."')
    if option['sampling']:
        parser.add_argument('sampleset', metavar='DS_NAME', type=str,
                            help='Name of sample dataset in active learning selecting algorithms')
        parser.add_argument('--load-state', action='store_true', default=False, help='Turn on if load state.')
        parser.add_argument('--state-suffix', metavar='SE', type=str,
                            help='load selected samples from sample set', required=False, default='')
    if option['synthetic']:
        parser.add_argument('synthetic_method', metavar='SM', type=str, help='Synthetic Method',
                            choices=['fgsm', 'ifgsm', 'mifgsm'])
        parser.add_argument('eps', metavar='E', type=float, help='Synthetic maximum epsilon')
        parser.add_argument('targeted_method', metavar='T', type=str, help='Target methods',
                            choices=['non-targeted', 'targeted-random', 'targeted-topk'])
    # if option['black_box']:
    parser.add_argument('victim_model_dir', metavar='VIC_DIR', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--pseudoblackbox', action='store_true', help='Load prequeried labels as blackbox',
                        default=False)
    parser.add_argument('--bydataset', action='store_true', help='Use dataset labels as blackbox', default=False)
    parser.add_argument('--topk', metavar='TK', type=int, help='iteration times',
                        default=0)
    if option['train']:
        parser.add_argument('model_dir', metavar='SUR_DIR', type=str, help='Surrogate Model Destination directory')
        parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
        parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
        # Optional arguments
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                            help='number of epochs to train (default: 100)')
        parser.add_argument('-x', '--complexity', type=int, default=64, metavar='N',
                            help="Model conv channel size.")
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.01)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                            help='SGD momentum (default: 0.5)')
        parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                            help='Step sizes for LR')
        parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                            help='LR Decay Rate')
        parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
        parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
        parser.add_argument('--optimizer-choice', type=str, help='Optimizer', default='sgdm',
                            choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    # apply to all circumstances
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-d', '--device-id', metavar='D', type=int, help='Device id. -1 for CPU.', default=7)
    parser.add_argument('-w', '--num-workers', metavar='N', type=int, help='# Worker threads to load data',
                        default=10)
    args = parser.parse_args()
    params = vars(args)
    print(params)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    device = torch.device('cuda')
    params['device'] = device
    if option['active']:
        pass
    if option['defense']:
        defense_type = params['defense']
        blackbox_dir = params['victim_model_dir']
        if defense_type == 'rand_noise':
            BB = RandomNoise
        elif defense_type == 'rand_noise_wb':
            BB = RandomNoise_WB
        elif defense_type == 'mad':
            BB = MAD
        elif defense_type == 'mad_wb':
            BB = MAD_WB
        elif defense_type == 'reverse_sigmoid':
            BB = ReverseSigmoid
        elif defense_type == 'reverse_sigmoid_wb':
            BB = ReverseSigmoid_WB
        elif defense_type in ['none', 'topk', 'rounding']:
            BB = Blackbox
        elif defense_type == 'edm':
            BB = EDM_device
        elif defense_type =='am':
            BB = AM
        else:
            raise ValueError('Unrecognized blackbox type')
        defense_kwargs = parse_defense_kwargs(params['defense_args'])
        defense_kwargs['log_prefix'] = 'transfer'
        print('=> Initializing BBox with defense {} and arguments: {}'.format(defense_type, defense_kwargs))
        params['blackbox'] = BB.from_modeldir(blackbox_dir, torch.device("cuda"), **defense_kwargs)
        for k, v in defense_kwargs.items():
            params[k] = v

    if option['sampling']:
        sample_set_name = params['sampleset']
        assert sample_set_name in datasets.__dict__.keys()
        modelfamily = datasets.dataset_to_modelfamily[sample_set_name]
        transform = datasets.modelfamily_to_transforms[modelfamily]['train']
        dataset = datasets.__dict__[sample_set_name](train=True, transform=transform)
        params['queryset'] = dataset
        params['selected'] = set()
        if params['load_state']:
            total = set([i for i in range(len(dataset))])
            path = params['model_dir']
            params['selection'], params['transferset'], params['selected_indices'] = load_state(path,
                                                                                                params['state_suffix'])
    if option['train']:
        testset_name = params['testdataset']
        assert testset_name in datasets.__dict__.keys()
        modelfamily = datasets.dataset_to_modelfamily[testset_name]
        transform = datasets.modelfamily_to_transforms[modelfamily]['test']
        testset = datasets.__dict__[testset_name](train=False, transform=transform)
        params['testset'] = testset

        pretrained_path = params['pretrained']
        model_arch = params['model_arch']
        if params['pseudoblackbox']:
            num_classes = params['blackbox'].train_results[0].shape[0]
        else:
            num_classes = len(testset.classes)
        sample = testset[0][0]

        model = zoo.get_net(model_arch, modelfamily, pretrained_path, num_classes=num_classes)
        params['surrogate'] = model.to(device)
    return params


def load_state(state_dir: str, selection_suffix: str = ''):
    transfer_path = os.path.join(state_dir, 'transferset.{}pickle'.format(selection_suffix))
    selection_path = os.path.join(state_dir, 'selection.{}pickle'.format(selection_suffix))
    indices_list_path = os.path.join(state_dir, 'select_indices.{}pickle'.format(selection_suffix))
    if not os.path.exists(transfer_path) or not os.path.exists(selection_path):
        print("State not exists, returning None")
        return set(), [], []
    with open(selection_path, 'rb') as sf:
        selection = pickle.load(sf)
        assert isinstance(selection, set)
        print("=> load selected {} sample indices from {}".format(len(selection), selection_path))
    with open(transfer_path, 'rb') as tf:
        transfer = pickle.load(tf)
        assert isinstance(transfer, List)
        print("=> load selected {} samples from {}".format(len(transfer), transfer_path))
    with open(indices_list_path, 'rb') as lf:
        indices_list = pickle.load(lf)
        assert isinstance(indices_list, List)
        print("=> load selected {} sample indices from {}".format(len(indices_list), indices_list_path))
    return selection, transfer, indices_list


def parse_defense_kwargs(kwargs_str):
    kwargs = dict()
    for entry in kwargs_str.split(','):
        if len(entry) < 1:
            continue
        key, value = entry.split(':')
        assert key not in kwargs, 'Argument ({}:{}) conflicts with ({}:{})'.format(key, value, key, kwargs[key])
        try:
            # Cast into int if possible
            value = int(value)
        except ValueError:
            try:
                # Try with float
                value = float(value)
            except ValueError:
                # Give up
                pass
        kwargs[key] = value
    return kwargs


def naive_onehot(index: int, total: int) -> Tensor:
    x = torch.zeros([total], dtype=torch.float32)
    x[index] = 1.0
    return x


class PseudoBlackbox(object):
    def __init__(self, target: Union[str, Dataset], argmax: bool = False):
        if isinstance(target, str):
            with open(os.path.join(target, 'train.pickle'), 'rb') as f:
                self.train_results = pickle.load(f)
            with open(os.path.join(target, 'eval.pickle'), 'rb') as f:
                eval_results = pickle.load(f)
            self.eval_results = [r.argmax() for r in eval_results]
            self.is_dataset = False
        elif isinstance(target, Dataset):
            self.train_results = target
            self.is_dataset = True
        self.argmax = argmax

    def __call__(self, index: int, train: bool = True):
        if self.is_dataset:
            x = self.train_results[index][1]
            x = naive_onehot(x, 3)
        else:
            x = self.train_results[index]

        if train:
            if self.argmax:
                temp = x
                m = temp.argmax()
                value = torch.zeros_like(temp)
                value[m] = 1.0
                return value
            return x
        else:
            return self.eval_results[index]


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, num_classes):
        self.dataset = dataset
        self.indices = indices
        self.num_classes = num_classes

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        y_v = torch.zeros([self.num_classes])
        y_v[y] = 1.0
        return x, y_v

    def __len__(self):
        return len(self.indices)


# todo this class should be rebuild on top of adversary
class ActiveAdversary(object):
    def __init__(self,
                 blackbox,
                 surrogate: Module,
                 queryset: Dataset,
                 testset: Dataset,
                 model_dir: str,
                 batch_size: int = 50,
                 num_workers: int = 15,
                 strategy: str = 'random',
                 metric: str = 'euclidean',
                 initial_size: int = 100,
                 device: torch.device = torch.device('cuda'),
                 optimizer_choice: str = 'sgdm',
                 **kwargs
                 ):
        self.device = device
        self.blackbox = blackbox
        self.surrogate = surrogate
        self.queryset = queryset
        self.path = model_dir
        self.kwargs = kwargs
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.batch_size = batch_size
        self.num_worker = num_workers

        self.evaluation_set: List[Tuple[Tensor, Tensor]] = testset
        assert strategy in ('random', 'kcenter')

        self.optimizer_choice = optimizer_choice
        self.optim = get_optimizer(self.surrogate.parameters(), optimizer_choice, **kwargs)

        self.criterion = model_utils.soft_cross_entropy
        # self.query_dataset([sample[0] for sample in testset], argmax=True, train=False)
        
        self.iterations = 0
        if kwargs.get('transferset'):
            self.selected = kwargs['transferset']
            self.queried = kwargs['selection']
            self.list_indices = kwargs['selected_indices']
            if strategy == 'random':
                self.sss = RandomSelectionStrategy(
                    dataset=self.queryset,
                    model=self.surrogate,
                    initial_size=initial_size,
                    seed=cfg.DEFAULT_SEED,
                    batch_size=self.batch_size
                )
            elif strategy == 'kcenter':
                self.sss = KCenterGreedyApproach(
                    dataset=self.queryset,
                    model=self.surrogate,
                    initial_size=initial_size,
                    seed=cfg.DEFAULT_SEED,
                    batch_size=self.batch_size,
                    metric=metric,
                    device=device,
                    initial_selection=self.list_indices
                )
            else:
                raise NotImplementedError
            print('selection: {}.'.format(len(self.sss.selected)))
            self.train()

        else:
            if strategy == 'random':
                self.sss = RandomSelectionStrategy(
                    dataset=self.queryset,
                    model=self.surrogate,
                    initial_size=initial_size,
                    seed=cfg.DEFAULT_SEED,
                    batch_size=self.batch_size
                )
            elif strategy == 'kcenter':
                self.sss = KCenterGreedyApproach(
                    dataset=self.queryset,
                    model=self.surrogate,
                    initial_size=initial_size,
                    seed=cfg.DEFAULT_SEED,
                    batch_size=self.batch_size,
                    metric=metric,
                    device=device
                )
            else:
                raise NotImplementedError
            self.selected: List[Tuple[Tensor, Tensor]] = []  # [(img_tensor, output_tensor)]
            # if self.blackbox.is_dataset:
            #     self.selected = Subset(self.queryset, [], 3)
            self.queried: Set[int] = set()
            self.query_index(self.sss.selecting)
            self.list_indices = list(self.sss.selecting)
            self.train()

    def query_dataset(self, training_samples: List[Tensor], argmax: bool = False, train: bool = True):
        with tqdm(total=len(training_samples)) as pbar:
            for t, B in enumerate(range(0, len(training_samples), self.batch_size)):
                x_t = torch.stack(
                    [training_samples[i] for i in range(B, min(B + self.batch_size, len(training_samples)))]).to(
                    self.device)
                y_t = self.blackbox(x_t)
                if self.kwargs['argmaxed'] or argmax:
                    y_t = y_t.argmax(1)
                elif self.kwargs['topk'] != 0:
                    v, i = y_t.topk(self.kwargs['topk'], 1)
                    y_t = torch.zeros_like(y_t).scatter(1, i, v)
                for i in range(x_t.size(0)):
                    if train:
                        self.selected.append((x_t[i].cpu(), y_t[i].cpu()))
                    else:
                        self.evaluation_set.append((x_t[i].cpu(), y_t[i].cpu()))
                pbar.update(x_t.size(0))

    def query_index(self, index_set: Set[int]):
        if len(index_set.intersection(self.queried)) > 0:
            raise Exception("Double query.")

        for index in index_set:
            x: Tensor = self.queryset[index][0].unsqueeze(0).to(self.device)
            y = self.blackbox(x)
            if self.kwargs['argmaxed']:
                y = y.argmax(1)
            elif self.kwargs['topk'] != 0:
                v, i = y.topk(self.kwargs['topk'], 1)
                y = torch.zeros_like(y).scatter(1, i, v)
            self.selected.append((x.squeeze(0).cpu(), y.squeeze(0).cpu()))
        self.queried.update(index_set)
        
        

    def train(self):
        # self.surrogate = zoo.get_net(self.kwargs["model_arch"], 'mnist', None, num_classes=10).to(self.device)
        # self.optim = get_optimizer(self.surrogate.parameters(), self.optimizer_choice, **self.kwargs)
        model_utils.train_model(self.surrogate, self.selected, self.path, batch_size=self.batch_size,
                                testset=self.evaluation_set, criterion_train=self.criterion,
                                checkpoint_suffix='.active.{}'.format(len(self.selected)), device=self.device,
                                restored=True, task='FashionMNIST',
                                optimizer=self.optim,
                                # **self.kwargs
                                )

    def save_selected(self):
        self.sss.merge_selection()
        selected_index_output_path = os.path.join(self.path, 'selection.pickle')
        selected_indices_list_path = os.path.join(self.path, 'select_indices.pickle')
        selected_transfer_outpath = os.path.join(self.path, "transferset.pickle")
        if os.path.exists(selected_index_output_path):
            print("{} exists, override file.".format(selected_index_output_path))
        with open(selected_index_output_path, 'wb') as fp:
            pickle.dump(self.sss.selected, fp)
        print("=> selected {} samples written to {}".format(len(self.sss.selected), selected_index_output_path))
        if os.path.exists(selected_indices_list_path):
            print("{} exists, override file.".format(selected_indices_list_path))
        with open(selected_indices_list_path, 'wb') as fp:
            pickle.dump(self.list_indices, fp)
        print("=> selected {} samples written to {}".format(len(self.list_indices), selected_indices_list_path))
        if os.path.exists(selected_transfer_outpath):
            print("{} exists, override file.".format(selected_transfer_outpath))
        with open(selected_transfer_outpath, 'wb') as fp:
            pickle.dump(self.selected, fp)
        print("=> selected {} samples written to {}".format(len(self.selected), selected_transfer_outpath))

    def step(self, size: int):
        self.sss.get_subset(size)
        self.list_indices.extend(self.sss.selecting)
        self.query_index(self.sss.selecting)
        self.train()
        self.iterations += 1


def main():
    torch.manual_seed(cfg.DEFAULT_SEED)
    params = parser_dealer(
        {
            'transfer': False,
            'active': True,
            'sampling': True,
            'synthetic': False,
            'black_box': False,
            'defense': True,
            'train': True
        }
    )

    active_adv = ActiveAdversary(**params)
    for i in range(params['iterations']):
        print("{} samples selected.".format(len(active_adv.sss.selected)))
        active_adv.step(params['budget_per_iter'])
        active_adv.save_selected()


if __name__ == '__main__':
    main()
