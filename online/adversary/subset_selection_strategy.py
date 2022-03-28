import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
from torch.nn import Module
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


class SubsetSelectionStrategy:
    """ Base Class of Active Learning Approach

    This is abstract class of Active Learning Approaches.
    """

    def __init__(self, dataset: Dataset, model: Module, seed: int = 1337, batch_size: int = 50,
                 device: torch.device = torch.device('cpu')) -> None:
        """ Initial algorithm.

        :param dataset: target dataset
        :param model: surrogate model
        :param seed: random seed for permutation
        :param batch_size: nn batch size
        """
        self.dataset = dataset
        self.model = model
        self.size = len(dataset)
        np.random.seed(seed)
        self.perm = np.random.permutation(self.size)
        self.perm_reverse = np.argsort(self.perm)
        self.state = np.random.get_state()
        self.selected = set()
        self.unselected = set([i for i in range(self.size)])
        self.selecting = set()
        self.batch_size = batch_size
        self.device = device

    def shuffle(self) -> None:
        """ shuffle index

        :return: None
        """
        np.random.set_state(self.state[0])
        self.perm = np.random.permutation(self.size)
        self.perm_reverse = np.argsort(self.perm)
        self.state = np.random.get_state()

    def query(self, x: List[int]) -> List[Tuple[int, Tensor]]:
        """ Query model

        :param x: indices to be queried
        :return: query_result, List[torch.Tensor]
        """
        x.sort()
        query_set = Subset(dataset=self.dataset, indices=x)
        loader = DataLoader(query_set, self.batch_size, shuffle=False)
        results = []
        for index, batch in enumerate(loader):
            result = self.model(batch)
            results.extend([
                (x[i], result[i])
                for i in range(index * self.batch_size, index * self.batch_size + result.shape[0])
            ])

        return results

    def merge_selection(self) -> None:
        """ Merge Selecting to Selected
        """
        if len(self.selecting) > 0:
            self.selected.update(self.selecting)
            self.selecting.clear()

    def select(self, index_list: List[int]) -> None:
        """ get selecting index into selecting set

        :param index_list: index of selecting samples
        """
        self.selecting.update(index_list)
        self.unselected.difference_update(index_list)

    def get_selecting_tensor(self) -> List[Tensor]:
        return [self.dataset[i][0] for i in self.selecting]

    def get_selected(self) -> (List[int], List[Tensor]):
        indexes = []
        tensors = []
        for index in self.selected:
            indexes.append(index)
            tensors.append(self.dataset[index][0])
        return indexes, tensors

    def query_all(self) -> List[Tensor]:
        loader = DataLoader(self.dataset, self.batch_size, shuffle=False)
        results = []
        for index, (x_batch, y_batch) in enumerate(loader):
            with torch.no_grad():
                result = self.model(x_batch.to(self.device)).cpu()
            results.extend([
                result[i]
                for i in range(result.shape[0])
            ])

        return results

    def get_subset(self, size: int) -> List[Tensor]:
        """ Abstract method which return the selection result

        :return: Selected tensor list
        """
        raise NotImplementedError


class RandomSelectionStrategy(SubsetSelectionStrategy):
    """ Random Selection

    For every get_subset operation, select random samples for `size`
    and set a selected records.
    """

    def __init__(self, dataset: Dataset, model: Module, initial_size: int = 1000, seed: int = 1337,
                 batch_size: int = 50) -> None:
        """ Initial algorithm.

        :param dataset: target dataset
        :param model: surrogate model
        :param seed: random seed for permutation
        :param batch_size: nn batch size
        """
        super(RandomSelectionStrategy, self).__init__(dataset, model, seed, batch_size)
        unselected_list = list(self.unselected)
        np.random.set_state(self.state)
        result_indexes = np.random.choice(unselected_list, initial_size, False)
        self.state = np.random.get_state()
        self.select(result_indexes)
        #     return self.get_selecting_tensor()

    def get_subset(self, size: int) -> set:
        self.merge_selection()
        unselected_list = list(self.unselected)
        np.random.set_state(self.state[0])
        result_indexes = np.random.choice(unselected_list, size, False)
        self.state = np.random.get_state()
        self.select(result_indexes)
        return self.selecting


class KCenterGreedyApproach(SubsetSelectionStrategy):
    """ K-Center Greedy Approach
    """

    def __init__(self,
                 dataset: Dataset,
                 model: Module,
                 initial_size: int,
                 seed: int = 1337,
                 batch_size: int = 50,
                 metric: str = 'euclidean',
                 device: torch.device = torch.device('cpu'),
                 initial_selection: List = None,
                 **kwargs
                 ):
        """ K Center Greedy Approach initialization

        :param dataset: target dataset
        :param model: surrogate model
        :param seed: random seed for permutation.
        :param batch_size: nn batch size
        :param metric: K-Center Approach distance metric. L1, L2 distance implemented.
        """
        assert metric in ('euclidean', 'manhattan', 'l1', 'l2')
        self.metric = metric
        super(KCenterGreedyApproach, self).__init__(dataset, model, seed, batch_size, device)
        if metric in ('euclidean', 'l2'):
            self.batch_size *= 4
        elif metric in ('manhattan', 'l1'):
            self.batch_size = int(self.batch_size / 8)
        unselected_list = list(self.unselected)
        if initial_selection is None:
            np.random.set_state(self.state)
            initial_selection = np.random.choice(unselected_list, initial_size, False)
            self.select(initial_selection)
            self.state = np.random.get_state()
        else:
            self.selected.update(initial_selection)
            self.unselected.difference_update(initial_selection)

    def get_subset(self, size: int) -> None:
        self.merge_selection()
        center_indexes, _ = self.get_selected()
        query_result = self.query_all()
        centers = [query_result[i] for i in center_indexes]
        for _ in tqdm(range(size), desc='Selecting'):
            min_distances = []
            min_indexes = []
            unselected_indexes = list(self.unselected)
            unselected_size = len(unselected_indexes)
            for batch_index, batch_initial in enumerate(range(0, unselected_size, self.batch_size)):
                torch_centers = torch.stack(centers)
                current_batch_indexes = unselected_indexes[batch_initial:batch_initial + self.batch_size]
                current_batch = torch.stack([query_result[i] for i in current_batch_indexes])
                min_dist, min_index = self.k_center(current_batch, torch_centers)
                min_distances.append(min_dist.cpu())
                min_indexes.append(unselected_indexes[batch_initial + min_index])
            selecting_i = min_indexes[int(np.argmax(min_distances))]
            centers.append(query_result[selecting_i])
            self.select([selecting_i])

        # return self.selecting

    def k_center(self, A: Tensor, B: Tensor) -> (Tensor, int):
        A = A.to(self.device)
        B = B.to(self.device)
        if self.metric == 'euclidean' or self.metric == 'l2':
            A_sq = torch.sum(torch.pow(A, 2), 1).reshape([-1, 1])
            B_sq = torch.sum(torch.pow(B, 2), 1).reshape([1, -1])
            dist = torch.sqrt(torch.max(A_sq - 2 * torch.matmul(A, B.T) + B_sq, torch.tensor(0.0).to(self.device)))
        elif self.metric == 'manhattan' or self.metric == 'l1':
            dist = torch.sum(torch.abs(A.unsqueeze(1) - B), -1)
        else:
            raise NotImplementedError
        min_dist, _ = torch.min(dist, dim=1)
        min_dist_max = torch.max(min_dist)
        min_dist_argmax = int(torch.argmax(min_dist))
        return min_dist_max, min_dist_argmax
