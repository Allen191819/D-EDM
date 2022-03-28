
import argparse
import os.path as osp
import os
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from online import datasets
import online.utils.utils as knockoff_utils
import online.config as cfg
from online.victim import *
from online.adversary.transfer import parse_defense_kwargs, BBOX_CHOICES
def test_step(adversary_model,
              victim_model,
              test_loader,
              device,
              epoch=0.,
              silent=False,
              writer=None):
    adversary_model.eval()
    victim_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_adversary = adversary_model(inputs)
            outputs_victim = victim_model(inputs)
            _, predicted_adversary = outputs_adversary.max(1)
            _, predicted_victim = outputs_victim.max(1)
            total += targets.size(0)
            correct += predicted_adversary.eq(predicted_victim).sum().item()

    agreement = 100. * correct / total

    if not silent:
        print('[Test]  Epoch: {}\tAgreement: {:.1f}% ({}/{})'.format(
            epoch, agreement, correct, total))

    if writer is not None:
        writer.add_scalar('Agreement/test', agreement, epoch)

    return agreement


def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument(
        'victim_model_dir',
        metavar='PATH',
        type=str,
        help=
        'Path to victim model. Should contain files "model_best.pth.tar" and "params.json"'
    )
    parser.add_argument(
        'adversary_model_dir',
        metavar="PATH",
        type=str,
        help=
        'Path to victim model. Should contain files "checkpoint.active.xx.pth.tar"'
    )
    parser.add_argument('budget',
                        metavar="N",
                        type=int,
                        help='Train budget size.')
    parser.add_argument('defense',
                        metavar='TYPE',
                        type=str,
                        help='Type of defense to use',
                        choices=BBOX_CHOICES)
    parser.add_argument('defense_args',
                        metavar='STR',
                        type=str,
                        help='Blackbox arguments in format "k1:v1,k2:v2,..."')
    parser.add_argument('--out_dir',
                        metavar='PATH',
                        type=str,
                        help='Destination directory to store transfer set',
                        required=True)
    parser.add_argument('--batch_size',
                        metavar='TYPE',
                        type=int,
                        help='Batch size of queries',
                        default=1)
    parser.add_argument('--topk',
                        metavar='N',
                        type=int,
                        help='Use posteriors only from topk classes',
                        default=None)
    parser.add_argument('--rounding',
                        metavar='N',
                        type=int,
                        help='Round posteriors to these many decimals',
                        default=None)
    # ----------- Other params
    parser.add_argument('-d',
                        '--device_id',
                        metavar='D',
                        type=int,
                        help='Device id',
                        default=0)
    parser.add_argument('-w',
                        '--nworkers',
                        metavar='N',
                        type=int,
                        help='# Worker threads to load data',
                        default=10)
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

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    defense_type = params['defense']

    adversary_model_dir = params['adversary_model_dir']
    budget = params['budget']
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
    else:
        raise ValueError('Unrecognized blackbox type')
    defense_kwargs = parse_defense_kwargs(params['defense_args'])
    defense_kwargs['log_prefix'] = 'test'
    print('=> Initializing BBox with defense {} and arguments: {}'.format(
        defense_type, defense_kwargs))
    blackbox = BB.from_modeldir(blackbox_dir, torch.device('cuda'),
                                **defense_kwargs)
    adversary_model = BB.get_adversary_model(blackbox_dir, adversary_model_dir,
                                             budget, torch.device('cuda'),
                                             **defense_kwargs)
    for k, v in defense_kwargs.items():
        params[k] = v

    # ----------- Set up queryset
    with open(osp.join(blackbox_dir, 'params.json'), 'r') as rf:
        bbox_params = json.load(rf)
    testset_name = bbox_params['dataset']
    valid_datasets = datasets.__dict__.keys()
    if testset_name not in valid_datasets:
        raise ValueError(
            'Dataset not found. Valid arguments = {}'.format(valid_datasets))

    modelfamily = datasets.dataset_to_modelfamily[testset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    testset = datasets.__dict__[testset_name](train=False, transform=transform)
    print('=> Evaluating on {} ({} samples)'.format(testset_name,
                                                    len(testset)))

    # ----------- Evaluate
    batch_size = params['batch_size']
    nworkers = params['nworkers']
    epoch = bbox_params['epochs']
    testloader = DataLoader(testset,
                            num_workers=nworkers,
                            shuffle=False,
                            batch_size=batch_size)
    agreement = test_step(adversary_model,
                          blackbox,
                          testloader,
                          device,
                          epoch=epoch)

    log_out_path = osp.join(
        out_path, 'adversaryeval.{}.B{}.log.tsv'.format(len(testset), budget))
    with open(log_out_path, 'w') as wf:
        columns = ['run_id', 'epoch', 'split', 'agreement']
        wf.write('\t'.join(columns) + '\n')

        run_id = str(datetime.now())
        test_cols = [run_id, epoch, 'test', agreement]
        wf.write('\t'.join([str(c) for c in test_cols]) + '\n')

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params_eval_adversary.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
