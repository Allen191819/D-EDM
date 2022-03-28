import os.path as osp
import time
from datetime import datetime
from collections import defaultdict as dd
import numpy as np
from offline.recovery import Restorer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as torch_models
from offline.utils_meta import load_model_setting
import online.utils.utils as knockoff_utils
from offline.meta_classifier import MetaClassifier


def get_net(model_name, n_output_classes=1000, **kwargs):
    print('=> loading model {} with arguments: {}'.format(model_name, kwargs))
    valid_models = [x for x in torch_models.__dict__.keys() if not x.startswith('__')]
    if model_name not in valid_models:
        raise ValueError('Model not found. Valid arguments = {}...'.format(valid_models))
    model = torch_models.__dict__[model_name](**kwargs)
    # Edit last FC layer to include n_output_classes
    if n_output_classes != 1000:
        if 'squeeze' in model_name:
            model.num_classes = n_output_classes
            model.classifier[1] = nn.Conv2d(512, n_output_classes, kernel_size=(1, 1))
        elif 'alexnet' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'vgg' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'dense' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, n_output_classes)
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_output_classes)
    return model


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))


class Resotrer(nn.Module):
    def __init__(self, num_classes=10):
        super(Resotrer, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, num_classes),
            nn.BatchNorm1d(num_classes),
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


def train_step(model, train_loader, criterion, optimizer, epoch, device, restored, task, log_interval=10,
               writer=None):
    model.train(),
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()
    GPU = True
    if task == 'CIFAR10':
        Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting("cifar10")
        generator_path = ''
        meta_path = ''
    elif task == 'MNIST':
        Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting("mnist")
        generator_path = ''
        meta_path = ''
    elif task == 'FashionMNIST':
        Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting("fashionmnist")
        generator_path = ''
        meta_path = ''
    elif task == 'CIFAR100':
        Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting("cifar100")
        generator_path = ''
        meta_path = ''
    elif task == 'ImageNette':
        Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting("imagenette")
        generator_path = ''
        meta_path = ''
    elif task == 'GTSRB':
        Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting("gtsrb")
        generator_path = ''
        meta_path = ''
    else:
        raise NotImplementedError("Unknown task %s" % task)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if restored:
            restorer = Restorer(num_classes=class_num).to(device)
            restorer.load_state_dict(torch.load(generator_path + '/Restorer.pth'))
            meta_classifier = MetaClassifier(input_size, class_num, N_in=1, gpu=GPU)
            meta_classifier.load_state_dict(torch.load(meta_path))
            for target in targets:
                score = meta_classifier.forward(target.float())
                score = F.sigmoid(score)
                targets = restorer(targets.float())
        else:
            pass
        optimizer.zero_grad()
        model.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total

        if (batch_idx + 1) % log_interval == 0:
            print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
                exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), acc, correct, total))

        if writer is not None:
            writer.add_scalar('Loss/train', loss.item(), exact_epoch)
            writer.add_scalar('Accuracy/train', acc, exact_epoch)

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, acc


def test_step(model, test_loader, criterion, device, epoch=0., silent=False, writer=None):
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            nclasses = outputs.size(1)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    acc = 100. * correct / total
    test_loss /= total

    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})'.format(epoch, test_loss, acc,
                                                                             correct, total))

    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)

    return test_loss, acc


def train_model(model, trainset, out_path, batch_size=32, criterion_train=None, criterion_test=None, testset=None,
                device=None, restored=False, task='MNIST', num_workers=5, lr=0.02, momentum=0.5, lr_step=30,
                lr_gamma=0.1,
                resume=None,
                epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None,
                scheduler=None,
                writer=None, **kwargs):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not osp.exists(out_path):
        knockoff_utils.create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)
    else:
        test_loader = None

    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, test_acc, test_loss = -1., -1., -1.

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
            wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device=device,
                                           restored=restored, task=task,
                                           log_interval=log_interval)
        scheduler.step(epoch)
        best_train_acc = max(best_train_acc, train_acc)

        if test_loader is not None:
            test_loss, test_acc = test_step(model, test_loader, criterion_test, device, epoch=epoch)
            best_test_acc = max(best_test_acc, test_acc)

        # Checkpoint
        if test_acc >= best_test_acc:
            state = {
                'epoch': epoch,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_acc': test_acc,
                'optimizer': optimizer.state_dict(),
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)

        # Log
        with open(log_path, 'a') as af:
            train_cols = [run_id, epoch, 'train', train_loss, train_acc, best_train_acc]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return model
