import numpy as np
import tarfile
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from online.victim.bb_reversesigmoid import ReverseSigmoid
from online.victim.bb_mad import MAD
from online.datasets.gtsrb import GTSRB
from online.datasets.mnistlike import MNIST, EMNIST, EMNISTLetters, FashionMNIST


def load_dataset_setting(task):
    if task == 'mnist':
        BATCH_SIZE = 100
        N_EPOCH = 40
        # TO DO: N_EPOCH has been changed to 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

        trainset = torchvision.datasets.MNIST(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./raw_data/', train=False, download=False, transform=transform)

        is_binary = False
        need_pad = False
        from model_lib.mnist_cnn_model import Model
    elif task == 'fashionmnist':
        BATCH_SIZE = 100
        N_EPOCH = 40
        # TO DO: N_EPOCH has been changed to 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

        trainset = torchvision.datasets.FashionMNIST(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./raw_data/', train=False, download=False,
                                                    transform=transform)

        is_binary = False
        need_pad = False
        from model_lib.mnist_cnn_model import Model
    elif task == 'cifar10':
        BATCH_SIZE = 100
        N_EPOCH = 100

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=False, transform=transform)
        is_binary = False
        need_pad = False
        from model_lib.cifar10_cnn_model import Model
    elif task == 'cifar100':
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
        trainset = torchvision.datasets.CIFAR100(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./raw_data/', train=False, download=False, transform=transform)
        is_binary = False
        need_pad = False
        from model_lib.cifar100_cnn_model import Model
    elif task == 'imagenette':
        BATCH_SIZE = 100
        N_EPOCH = 100

        stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz'
        Database = download_url(dataset_url, './raw_data/')

        with tarfile.open('./raw_data/imagenette-160.tgz', 'r:gz') as tar:
            tar.extractall(path='../raw_data')
        data_dir = './raw_data/imagenette-160'

        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats, inplace=True)
        ])

        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])

        trainset = ImageFolder(data_dir + '/train', train_transform)
        testset = ImageFolder(data_dir + '/val', test_transform)
        is_binary = False
        need_pad = False
        from model_lib.imagenette_cnn_model import Model
    elif task == 'gtsrb':
        BATCH_SIZE = 100
        N_EPOCH = 100
        stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats, inplace=True)
        ])

        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])

        trainset = GTSRB(train=True, transform=train_transform)
        testset = GTSRB(train=False, transform=test_transform)

        is_binary = False
        need_pad = False
        from model_lib.gtsrb_cnn_model import Model

    elif task == 'emnistletters':
        BATCH_SIZE = 100
        N_EPOCH = 40
        # TO DO: N_EPOCH has been changed to 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

        trainset = EMNISTLetters(train=True, transform=transform)
        testset = EMNISTLetters(train=True, transform=transform)

        is_binary = False
        need_pad = False
        from model_lib.mnist_cnn_model import Model

    elif task == 'emnist':
        BATCH_SIZE = 100
        N_EPOCH = 40
        # TO DO: N_EPOCH has been changed to 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

        trainset = EMNIST(train=True, transform=transform)
        testset = EMNIST(train=True, transform=transform)

        is_binary = False
        need_pad = False
        from model_lib.mnist_cnn_model import Model
    elif task == 'lisa':
        pass
    elif task == 'miniimagenet':
        pass
    else:
        raise NotImplementedError("Unknown task %s" % task)

    return BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, need_pad, Model


def compute_noise(y, strat, epsilon_z, zdist):
    """
        Compute noise in the logit space (inverse sigmoid)
        :param y:
        :return:
        """
    z = ReverseSigmoid.inv_sigmoid(y)
    N, K = z.shape
    if strat == 'uniform':
        deltaz = torch.rand_like(z)

        # Norm of expected value of this distribution (|| E_{v ~ Unif[0, high]^K}[v] ||_p) is:
        #       \sqrt[p]{K} * (high=1)/2
        # Setting this to epsilon and solving for high', we get: high' = (2 * epsilon) / \sqrt[p]{K}
        # By drawing a k-dim vector v uniformly in the range [0, high'], we get || E[v] ||_p = epsilon
        if zdist in ['l1', 'l2']:
            p = int(zdist[-1])
            mult = (2 * epsilon_z) / np.power(K, 1. / p)
            # Rescale to [0, high']
            deltaz *= mult
    elif strat == 'gaussian':
        deltaz = torch.randn_like(z)
    else:
        raise ValueError('Unrecognized argument')

    for i in range(N):
        # Project each delta back into ydist space
        # print('Before: {} (norm-{} = {})'.format(deltaz[i], zdist, deltaz[i].norm(p=int(zdist[-1]))))
        deltaz[i] = MAD.project_ydist_constraint(deltaz[i], epsilon_z, zdist)
        # print('After: {} (norm-{} = {})'.format(deltaz[i], zdist, deltaz[i].norm(p=int(zdist[-1]))))
        # print()

    ztilde = z + deltaz
    ytilde = torch.sigmoid(ztilde)
    if len(ytilde.shape) > 1:
        ytilde /= ytilde.sum(dim=1)[:, None]
    else:
        ytilde = ytilde / ytilde.sum()
    delta = ytilde - y

    return delta


def train_model(model, dataloader, epoch_num, is_binary, verbose=True, defense=None):
    model.train(),
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epoch_num):
        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0
        for i, (x_in, y_in) in enumerate(dataloader):
            x = x_in.to(device)
            y = y_in.to(device)
            B = x_in.size()[0]
            pred = model(x)
            loss = model.loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss.item() * B

            if is_binary:
                cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
            else:
                pred_c = pred.max(1)[1].cpu()
                cum_acc += (pred_c.eq(y_in)).sum().item()
            tot = tot + B
        if verbose:
            print("Epoch %d, loss = %.4f, acc = %.4f" % (epoch, cum_loss / tot, cum_acc / tot))

    return


def sigmoid(var_z):
    return 1.0 / (1.0 + torch.exp(-var_z))


def eval_model(model, dataloader, is_binary, defense=None):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, (x_in, y_in) in enumerate(dataloader):
        x = x_in.to(device)
        y = y_in.to(device)
        B = x_in.size()[0]
        pred = model(x)
        if defense is None:
            pred = pred
        elif defense == 'ReverseSigmoid':
            clip_min = 1e-9
            clip_max = 1.0 - clip_min
            beta = 1.0
            gamma = 0.1
            pred = pred - ReverseSigmoid.reverse_sigmoid(pred, beta, gamma)
            pred /= pred.sum(dim=1)[:, None]
        elif defense == 'HighConfidence':
            cutoff = 0.25
            pred[pred < cutoff] = 0.0
        elif defense == 'GaussianNoise':
            scale = 0.2
            epsilonz = 0.2
            ydist = 'l1'
            strat = 'gaussian'
            pred = F.softmax(pred, dim=1).detach()
            noise = compute_noise(pred, strat, epsilonz, ydist).to('CPU')
            pred += noise
        elif defense == 'ClassLabels':
            class_labels = torch.zeros_like(pred)
            index_labels = torch.argmax(class_labels)
            class_labels[:, index_labels] = 1
            pred = class_labels
        elif defense == 'Rounding':
            pred = torch.round(pred * 10 ** 2) / (10 ** 2)
        else:
            raise NotImplementedError("Unknown defense %s" % defense)
        if is_binary:
            cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
        else:
            pred_c = pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot
