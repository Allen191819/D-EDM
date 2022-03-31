import os.path
from typing import List
from online.victim import Blackbox
import torch
import torch.nn.functional as F
from online.victim.utils_EDM import get_model
cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

model_choices = {
    'mnist':('lenet','conv3', 'kmnist'),
    'fashion':('lenet', 'conv3', 'kmnist'),
    'cifar10':('vgg16_bn', 'conv3', 'cifar100'),
    'gtsrb':('resnet34', 'conv3', 'lisa'),
    'imagenette':('resnet34', 'conv3', 'miniimagenet')
}

class EDM_device(Blackbox):
    def __init__(
        self,
        root = '',
        task = ''
    ):
        self.root = root
        self.call_count=0
        self.dataset_tar = task
        self.model_tar, self.arch_hash, self.hash_ds = model_choices[task]
        self.device = torch.device('cuda:1')
        self.model_list = self.load_models()
        self.n_models = len(self.model_list)
        for model in self.model_list:
            model = model.to(self.device)
            model.eval()
        self.bounds = [-1, 1]
        self.n_queries = 0
        self.model_hash = self.load_hash_model()
        self.hash_list = []

        if self.model_hash is not None:
            self.model_hash = self.model_hash.to(self.device)

    @classmethod
    def from_modeldir(cls, model_dir, device, output_type='probs', **kwargs):
        return cls(task=kwargs['task'])

    def load_models(self) -> List:
        T_list = []
        path_exp = os.path.join('', f'{self.dataset_tar}/alpha')
        for i in range(5):
            T_path = os.path.join(path_exp, 'T' + str(i) + '.pt')
            print("**",self.dataset_tar)
            T = get_model(self.model_tar, self.dataset_tar).to(self.device)
            T.load_state_dict(torch.load(T_path))
            T_list.append(T)

        return T_list

    def load_hash_model(self):
        path_hash = os.path.join('', f'{self.hash_ds}/hash.pt')
        #print(path_hash)
        H = get_model(self.arch_hash, self.hash_ds)
        H.load_state_dict(torch.load(path_hash))
        return H

    def eval(self):
        for model in self.model_list:
            model.eval()

    def coherence(self, x):
        pred_list = []
        cs_list = []
        for model in self.model_list:
            pred = F.softmax(model(x), dim=-1)
            pred_list.append(pred)
        pred_list_batch = torch.stack(pred_list, dim=0)  # n x batch x 10

        for i, pred_i in enumerate(pred_list_batch):
            for j in range(i + 1, len(self.model_list)):
                pred_j = pred_list_batch[j]
                cs = cosine_similarity(pred_i, pred_j)  # batch x 10
                cs_list.append(cs.detach())

        cs_batch = torch.max(torch.stack(cs_list, dim=0), dim=0)[0]
        return cs_batch

    def to(self, device: str):
        for model in self.model_list:
            model = model.to(device)
        return self

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.round(x * 128.0) / 128.0
        return x

    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        """ clamp """
        x = torch.clamp(x, self.bounds[0], self.bounds[1])
        return x


    def get_hash_list(self, x: torch.Tensor) -> List[int]:
        y = F.softmax(self.model_hash(x), dim=-1)
        y_class = torch.argmax(y, dim=-1)
        M = 10
        m = 2
        y_hash = (
            (m * y_class) + torch.floor((M * torch.max(y, dim=-1)[0] - 1) * m / (M - 1))
        ) % self.n_models
        return y_hash.detach().cpu().numpy().tolist()


    def update_stats(self, hash_list):
        self.hash_list += hash_list

    def __call__(self, x: torch.Tensor,) -> torch.Tensor:
        self.n_queries += x.shape[0]
        self.call_count += x.shape[0]
        out_list = []
        
        for model in self.model_list:
            with torch.no_grad():
                out = model(x)
                out_list.append(out)
                out = F.softmax(out,dim=1).detach()
        out_all = torch.stack(out_list, dim=0)
        hash_list = self.get_hash_list(x)
        out = out_all[hash_list, range(x.shape[0])]

        self.update_stats(hash_list)

        return out

    def get_n_queries(self) -> int:
        return self.n_queries

