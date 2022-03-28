import os.path as osp
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from online.utils.type_checks import TypeCheck
from online.victim import Blackbox, MAD
np.seterr(invalid='ignore')


class ReverseSigmoid_device():

    def __init__(self, model, beta=1.0, gamma=1.0, out_path=None, log_prefix='',
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        print('=> ReverseSigmoid ({})'.format([beta, gamma]))

        self.call_count = 0
        self.model = model
        self.device = device
        assert beta >= 0.
        assert gamma >= 0.
        self.out_path = out_path

        self.beta = beta
        self.gamma = gamma

        # Track some data for debugging
        self.queries = []  # List of (x_i, y_i, y_i_prime, distance)
        self.log_path = osp.join(out_path, 'distance{}.log.tsv'.format(log_prefix))
        if not osp.exists(self.log_path):
            with open(self.log_path, 'w') as wf:
                columns = ['call_count', 'l1_mean', 'l1_std', 'l2_mean', 'l2_std', 'kl_mean', 'kl_std']
                wf.write('\t'.join(columns) + '\n')

    @staticmethod
    def sigmoid(z):
        return torch.sigmoid(z)

    @staticmethod
    def inv_sigmoid(p):
        assert (p >= 0.).any()
        assert (p <= 1.).any()
        return torch.log(p / (1 - p))

    @staticmethod
    def reverse_sigmoid(y, beta, gamma):
        """
        Equation (3)
        :param y:
        :return:
        """
        return beta * (ReverseSigmoid_device.sigmoid(gamma * ReverseSigmoid_device.inv_sigmoid(y)) - 0.5)

    def __call__(self, x):
        TypeCheck.multiple_image_blackbox_input_tensor(x)  # of shape B x C x H x W

        with torch.no_grad():
            x = x.to(self.device)
            z_v = self.model(x)  # Victim's predicted logits
            y_v = F.softmax(z_v, dim=1)
            self.call_count += x.shape[0]

        # Inner term of Equation 4
        y_prime = y_v - ReverseSigmoid_device.reverse_sigmoid(y_v, self.beta, self.gamma)

        # Sum to 1 normalizer "alpha"
        y_prime /= y_prime.sum(dim=1)[:, None]

        for i in range(x.shape[0]):
            self.queries.append((y_v[i].cpu().detach().numpy(), y_prime[i].cpu().detach().numpy()))

            if (self.call_count + i) % 1000 == 0:
                # Dump queries
                query_out_path = osp.join(self.out_path, 'queries.pickle')
                with open(query_out_path, 'wb') as wf:
                    pickle.dump(self.queries, wf)

                l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = MAD.calc_query_distances(self.queries)

                # Logs
                with open(self.log_path, 'a') as af:
                    test_cols = [self.call_count, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

        return y_prime
