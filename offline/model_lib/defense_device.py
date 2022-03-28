import numpy as np
import torch
import torch.nn.functional as F
from online.victim.bb_mad_detached import MAD_device
from online.victim.bb_reversesigmoid_detached import ReverseSigmoid_device
from online.victim.bb_EDM_detached import EDM_device
from online.victim.bb_AM_detached import AM_device
from offline.utils_basic import compute_noise
from torchvision import transforms


class Victim:

    def __init__(self, model=None, gpu=False):
        self.model = model
        self.device = torch.device("cuda") if gpu else torch.device("cpu")

        pass

    def __call__(self, x, defense=None):
        x_tmp = x.clone().detach().to(self.device)
        y = self.model(x).to(self.device)

        if defense is None:
            y = F.softmax(y, dim=1)
            return y
        elif defense == 'HighConfidence':
            cutoff = 0.25
            y[y < cutoff] = 0.0
            y = F.softmax(y, dim=1)
            return y
        elif defense == 'GaussianNoise':
            y = F.softmax(y, dim=1).detach()
            epsilonz = 0.2
            ydist = 'l1'
            strat = 'gaussian'
            noise = compute_noise(y, strat, epsilonz, ydist).to('cuda')
            y += noise
            y = F.softmax(y, dim=1)
            return y
        elif defense == 'ClassLabels':
            class_labels = torch.zeros_like(y)
            index_labels = torch.argmax(class_labels)
            class_labels[:, index_labels] = 1
            y = class_labels
            y = F.softmax(y, dim=1)
            return y
        elif defense == 'Rounding':
            y = torch.round(y * 10 ** 2) / (10 ** 2)
            return y
        elif defense == 'AM':
            self.AM_bb = AM_device(self.model,
                                   delta_list=[0.99], # comma separated values specifying defense levels: delta(SM)
                                   num_classes=10)
            y_mod = self.AM_bb(x, y)
            y_mod = F.softmax(y_mod, dim=1).detach()
            return y_mod
        elif defense == 'EDM':
            pass
            if not hasattr(self, 'EDM_bb'):
                num_classes = y.shape[1]
                self.dataset_tar = 'gtsrb'
                # TODO: change dataset name
                self.EDM_bb = EDM_device(model=self.model, task=self.dataset_tar)
            EDM_device.mode = self.model.to(self.device)
            hash_list = []
            x_mod = EDM_perturb(x)
            hash_mod = np.array(self.EDM_bb.get_hash_list(x_mod))
            hash_list.append(hash_mod)
            y_prime = self.EDM_bb(x_mod)
            y_prime = F.softmax(y_prime, dim=1).detach()
            y_prime /= y_prime.sum(dim=1)[:, None]
            return y_prime

        elif defense == 'ReverseSigmoid':
            # TODO: perturb the output using Reverse Sigmoid
            pass
            if not hasattr(self, 'ReverseSigmoid_bb'):
                num_classes = x.shape[1]
                self.beta = 0.3
                self.gamma = 0.2
                self.ReverseSigmoid_bb = ReverseSigmoid_device(self.model, beta=self.beta, gamma=self.gamma,
                                                               out_path='models/final_bb_dist/ImageNette-resnet34-reverse_sigmoid-beta0.1-gamma0.2-ImageNette-B10000-proxy_scratch-random'
                                                               # out_path='models/final_bb_dist/CIFAR100-vgg16_bn-reverse_sigmoid-beta0.1-gamma0.1-CIFAR100-B60000-proxy_scratch-random'
                                                               )

            ReverseSigmoid_device.mode = self.model.to(self.device)
            y_v = F.softmax(y, dim=1).detach()
            y_prime = y_v - self.ReverseSigmoid_bb.reverse_sigmoid(y_v, beta=self.beta, gamma=self.gamma)
            y_prime /= y_prime.sum(dim=1)[:, None]
            return y_prime

        elif defense == 'MAD':
            pass
            # TODO: pertube the output using Prediction Poisoning
            # CVPR 2021, "Prediction Poisoning: Towards Defenses Against DNN Model Stealing Attacks"
            # refer to 'https://arxiv.org/abs/1906.10908'

            if not hasattr(self, 'MAD_bb'):
                num_classes = y.shape[1]
                epsilon = 0.5
                optim = 'linesearch'  # Choice from ['linesearch', 'projections', 'greedy']
                model_adv_proxy = None
                max_grad_layer = None  # Choice from [None, 'all']
                objmax = True
                ydist = 'l1'   # Choice from ['l1', 'l2', 'kl']
                oracle = 'random'  # Choice from ['extreme', 'random', 'argmin', 'argmax']
                disable_jacobian = False
                self.MAD_bb = MAD_device(self.model, epsilon=epsilon, optim=optim,
                                         model_adv_proxy='models/victim/MNIST-lenet-train-nodefense-scratch-advproxy',
                                         # model_adv_proxy='models/victim/FashionMNIST-lenet-train-nodefense-scratch-advproxy',
                                         # model_adv_proxy='models/victim/CIFAR100-vgg16_bn-train-nodefense-scratch-advproxy',
                                         # model_adv_proxy='models/victim/CIFAR10-vgg16_bn-train-nodefense-scratch-advproxy',
                                         # model_adv_proxy='models/victim/ImageNette-resnet34-train-nodefense-scratch-advproxy',
                                         # model_adv_proxy='models/victim/GTSRB-vgg16_bn-train-nodefense-scratch-advproxy',
                                         max_grad_layer=max_grad_layer, objmax=objmax, ydist=ydist,
                                         num_classes=num_classes, oracle=oracle, disable_jacobian=disable_jacobian,
                                         # out_path='models/final_bb_dist/FashionMNIST-lenet-mad_l1-eps0.5-EMNIST-B60000-proxy_scratch-random',
                                         # out_path='models/final_bb_dist/MNIST-lenet-mad_l1-eps0.5-EMNISTLetters-B60000-proxy_scratch-random',
                                         out_path='models/final_bb_dist/GTSRB-vgg16_bn-mad_l1-eps0.5-GTSRB-B30000-proxy_scratch-random',
                                         # out_path='models/final_bb_dist/CIFAR10-vgg16_bn-mad_l1-eps0.5-CIFAR10-B60000-proxy_scratch-random',
                                         # out_path='models/final_bb_dist/CIFAR100-vgg16_bn-mad_l1-eps0.5-CIFAR10-B60000-proxy_scratch-random',
                                         # out_path='models/final_bb_dist/ImageNette-resnet34-mad_eps0.5-ImageNette-B10000-proxy_scratch-random',
                                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            MAD_device.model = self.model.to(self.device)
            y_prime = []
            y_v = F.softmax(y, dim=1).detach()
            # No batch support yet. So, perturb individually.
            for i in range(y.shape[0]):
                x_i = x_tmp[i].unsqueeze(0).to(self.device)
                y_v_i = y_v[i].to(self.device)
                y_prime_i = self.MAD_bb.perturb(x_i, y_v_i)
                y_prime.append(y_prime_i)
            y_prime = torch.stack(y_prime)
            return y_prime
        else:
            raise NotImplementedError("Unknown defense %s" % defense)



def EDM_perturb(x_batch, bounds=[-1, 1]):
    x_batch = (x_batch - bounds[0]) / (bounds[1] - bounds[0])
    x_batch = x_batch.cpu()

    if x_batch.ndim == 3:
        x_batch = x_batch.unsqueeze(0)

    if x_batch.shape[1] == 1:
        normalize = transforms.Normalize([0.5], [0.5])
    else:
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    data_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                15, translate=(0.1, 0.1), scale=(0.9, 1.0), shear=(0.1, 0.1)
            ),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    '''
    data_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                25, translate=(0.2, 0.2), scale=(0.8, 1.0), shear=(0.1, 0.1)
            ),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    '''
    x_batch_mod = torch.stack([data_transforms(xi) for xi in x_batch], axis=0)
    device = torch.device("cuda")
    x_batch_mod = x_batch_mod.to(device)

    return x_batch_mod

