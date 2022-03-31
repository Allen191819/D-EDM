from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import os.path as osp
import random
import numpy as np
from utils_basic import load_dataset_setting
from model_lib.defense_device import Victim
import online.utils.utils as knockoff_utils


def train(epoch_itrs, batch_size, nz, shadow, generator, restorer, device, optimizer, epoch, shadow_loader,
          verbose=False,
          defense=None):
    generator.train(),
    restorer.train(),
    optimizer_R, optimizer_G = optimizer

    for i in range(epoch_itrs):
        for j in range(len(shadow)):
            for k, (data, target) in enumerate(shadow_loader):
                data, target = data.to(device), target.to(device)
                optimizer_R.zero_grad()
                t_logit = shadow[j](data, defense=None)
                s_logit = shadow[j](data, defense=defense)
                r_logit = restorer(s_logit.detach())
                loss_S = F.l1_loss(r_logit, t_logit.detach())


                loss_S.backward()
                optimizer_R.step()
        if verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t S_loss: {:.6f}'.format(
                epoch, i, epoch_itrs, 100 * float(i) / float(epoch_itrs), loss_S.item()))


class Restorer(nn.Module):
    def __init__(self):
        super(Restorer, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(10, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 10),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh(),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output

    def predict(self, input):
        output = self.main(input)
        output /= output.sum(dim=1)[:, None]
        return output


def test(nz, restorer, generator, shadow, device, test_loader, epoch, defense):
    restorer.eval()
    generator.eval()

    test_loss_defensed = 0
    test_loss_restored = 0
    dist_defensed_l2 = 0
    dist_restored_l2 = 0
    dist_defensed_l1 = 0
    dist_restored_l1 = 0
    correct_restored = 0
    correct_benign = 0
    correct_defensed = 0
    restore_identity = 0
    defensed_identity = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            for j in range(len(shadow)):
                shadow[j].model.eval()
                output_benign = shadow[j](data, defense=None)
                output_defensed = shadow[j](data, defense=defense)
                output_restored = restorer(output_defensed)
                test_loss_defensed += F.cross_entropy(output_defensed, target, reduction='sum').item()
                test_loss_restored += F.cross_entropy(output_restored, target, reduction='sum').item()
                dist_defensed_l2 += 0
                dist_restored_l2 += 0
                dist_defensed_l1 += 0
                dist_restored_l1 += 0

                pred_defensed = output_defensed.argmax(dim=1, keepdim=True)
                pred_restored = output_restored.argmax(dim=1, keepdim=True)
                pred_benign = output_benign.argmax(dim=1, keepdim=True)

                restore_identity += pred_benign.eq(pred_restored).sum().item()
                defensed_identity += pred_benign.eq(pred_defensed).sum().item()

                correct_restored += pred_restored.eq(target.view_as(pred_restored)).sum().item()
                correct_defensed += pred_defensed.eq(target.view_as(pred_defensed)).sum().item()
                correct_benign += pred_benign.eq(target.view_as(pred_benign)).sum().item()

    test_loss_defensed /= (len(test_loader.dataset) * len(shadow))
    test_loss_restored /= (len(test_loader.dataset) * len(shadow))
    dist_defensed_l2 /= (len(test_loader.dataset) * len(shadow))
    dist_defensed_l1 /= (len(test_loader.dataset) * len(shadow))
    dist_restored_l2 /= (len(test_loader.dataset) * len(shadow))
    dist_restored_l1 /= (len(test_loader.dataset) * len(shadow))

    print('Epoch {} Test set: Average loss: {:.4f}, Restored Accuracy: {}/{} ({:.4f}%)\n'.format(
        epoch, test_loss_restored, correct_restored, len(test_loader.dataset) * len(shadow),
                                                     100. * (correct_restored / (
                                                             len(test_loader.dataset) * len(shadow)))))
    print('Defensed Accuracy: {}/{} ({:.4f}%)\n'.format(
        correct_defensed, len(test_loader.dataset) * len(shadow),
                          100. * (correct_defensed / (len(test_loader.dataset) * len(shadow)))))
    print('Benign Accuracy: {}/{} ({:.4f}%)\n'.format(
        correct_benign, len(test_loader.dataset) * len(shadow),
                        100. * (correct_benign / (len(test_loader.dataset) * len(shadow)))))

    print('Restorer Performance: {}/{} ({:.4f}%)\n'.format(
        restore_identity, len(test_loader.dataset) * len(shadow),
                          100. * (restore_identity / (len(test_loader.dataset) * len(shadow)))))

    print('Defense Performance: {}/{} ({:.4f}%)\n'.format(
        defensed_identity, len(test_loader.dataset) * len(shadow),
                           100. * (defensed_identity / (len(test_loader.dataset) * len(shadow)))))

    acc_benign = correct_benign / (len(test_loader.dataset) * len(shadow))
    acc_defensed = correct_defensed / (len(test_loader.dataset) * len(shadow))
    acc_restored = correct_restored / (len(test_loader.dataset) * len(shadow))
    return acc_benign, acc_defensed, acc_restored, test_loss_defensed, test_loss_restored, dist_restored_l1, dist_restored_l2, dist_defensed_l1, dist_defensed_l2


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train restorer')
    parser.add_argument('--task', type=str, default='mnist',
                        help='Specfiy the task (mnist/cifar10/gtsrb/imagenette/fashionmnist).')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 16)')
    parser.add_argument('--lr_S', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=0.0002,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epoch_itrs', type=int, default=20)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--step_size', type=int, default=100, metavar='S')
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--defense', type=str, default='MAD', help='Specify which defense is used('
                                                                   'ReverseSigmoid/ClassLabels/HighConfidence'
                                                                   '/GaussianNoise/Rounding/MAD/AM/EDM).')
    args = parser.parse_args()
    GPU = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('restorer', exist_ok=True)
    # checkpoint/student
    print(args)

    SHADOW_NUM = 2048 + 256
    num_shadow = 4  # shadow model used to train GAN in one epoch
    SHADOW_PROP = 0.02
    save_path_G = './generator/%s' % args.task + '/%s' % args.defense
    np.random.seed(0)
    torch.manual_seed(0)
    shadow_path = './shadow/%s/models' % args.task
    BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, _, Model = load_dataset_setting(args.task)

    tot_num = len(trainset)
    shadow_indices = np.random.choice(tot_num, int(tot_num * SHADOW_PROP))
    print("Data indices owned by the defender:", shadow_indices)
    shadow_set = torch.utils.data.Subset(trainset, shadow_indices)
    shadow_loader = torch.utils.data.DataLoader(shadow_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    shadow_model_set = []
    for i in range(SHADOW_NUM):
        x = shadow_path + '/shadow_%d.model' % i
        shadow_model_set.append(x)

    restorer = Restorer()
    restorer = restorer.to(device)

    optimizer_R = optim.SGD(restorer.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)

    if args.scheduler:
        scheduler_R = optim.lr_scheduler.StepLR(optimizer_R, args.step_size, 0.1)

    acc_list = []

    # Initialize logging
    if not osp.exists(save_path_G):
        knockoff_utils.create_dir(save_path_G)
    log_path = osp.join(save_path_G, 'train{}.log.tsv'.format(''))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['epoch', 'acc_benign', 'acc_defensed', 'acc_restored', 'l1_dist', 'l2_dist', 'loss']
            wf.write('\t'.join(columns) + '\n')

    for epoch in range(1, args.epoch_itrs + 1):
        # Train
        shadow = []

        perm = np.random.choice(len(shadow_model_set), num_shadow)
        for j in perm:
            x = shadow_model_set[j]
            shadow_model = Model(gpu=GPU)

            shadow_model.load_state_dict(torch.load(x))
            shadow_model.eval()
            shadow_model = shadow_model.to(device)
            bb = Victim(gpu=GPU)
            bb.model = shadow_model
            shadow.append(bb)

        if args.scheduler:
            scheduler_R.step()

        train(N_EPOCH, 4, args.nz, shadow,
              restorer, restorer, device, optimizer=[optimizer_R, optimizer_R], epoch=epoch,
              shadow_loader=shadow_loader,
              verbose=False,
              defense=args.defense)

        # Test
        acc_benign, acc_defensed, \
        acc_restored, test_loss_defensed, \
        test_loss_restored, dist_restored_l1, \
        dist_restored_l2, dist_defensed_l1, \
        dist_defensed_l2 = test(args.nz, restorer, restorer,
                                shadow, device, test_loader, epoch=0,
                                defense=args.defense)
        acc_list.append(acc_restored)
        # Log
        with open(log_path, 'a') as af:
            restore_cols = [epoch, acc_benign, acc_defensed, acc_restored, dist_restored_l1, dist_restored_l2,
                            test_loss_restored]
            af.write('\t'.join([str(c) for c in restore_cols]) + '\n')

    torch.save(restorer.state_dict(), save_path_G + '/Restorer.pth')


if __name__ == '__main__':
    main()
