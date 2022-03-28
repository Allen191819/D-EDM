import numpy as np
import torch
from offline.utils_basic import load_dataset_setting, train_model
from sklearn.metrics import roc_auc_score
import os


def load_model_setting(task):
    if task == 'mnist':
        from model_lib.mnist_cnn_model import Model
        input_size = (1, 28, 28)
        class_num = 10
        normed_mean = np.array((0.1307,))
        normed_std = np.array((0.3081,))
        is_discrete = False
    elif task == 'fashionmnist':
        from model_lib.mnist_cnn_model import Model
        input_size = (1, 28, 28)
        class_num = 10
        normed_mean = np.array((0.1307,))
        normed_std = np.array((0.3081,))
        is_discrete = False
    elif task == 'cifar10':
        from model_lib.cifar10_cnn_model import Model
        input_size = (3, 32, 32)
        class_num = 10
        normed_mean = np.reshape(np.array((0.4914, 0.4822, 0.4465)), (3, 1, 1))
        normed_std = np.reshape(np.array((0.2023, 0.1994, 0.2010)), (3, 1, 1))
        is_discrete = False
    elif task == 'cifar100':
        from model_lib.cifar100_cnn_model import Model
        input_size = (3, 32, 32)
        class_num = 100
        normed_mean = np.reshape(np.array((0.4914, 0.4822, 0.4465)), (3, 1, 1))
        normed_std = np.reshape(np.array((0.2023, 0.1994, 0.2010)), (3, 1, 1))
        is_discrete = False
    elif task == 'imagenette':
        from model_lib.imagenette_cnn_model import Model
        input_size = (3, 64, 64)
        class_num = 10
        normed_mean = np.reshape(np.array((0.485, 0.456, 0.406)), (3, 1, 1))
        normed_std = np.reshape(np.array((0.229, 0.224, 0.225)), (3, 1, 1))
        is_discrete = False
    elif task == 'gtsrb':
        from model_lib.gtsrb_cnn_model import Model
        input_size = (3, 32, 32)
        class_num = 43
        normed_mean = np.reshape(np.array((0.485, 0.456, 0.406)), (3, 1, 1))
        normed_std = np.reshape(np.array((0.229, 0.224, 0.225)), (3, 1, 1))
        is_discrete = False
    else:
        raise NotImplementedError("Unknown task %s" % task)

    return Model, input_size, class_num, normed_mean, normed_std, is_discrete


def epoch_meta_train(bb, meta_model, basic_model, optimizer, dataset, is_discrete, threshold=0.0,
                     defense=None, task=None):
    meta_model.train(),
    basic_model.train(),

    cum_loss = 0.0
    preds = []
    labs = []
    perm = np.random.permutation(len(dataset))
    for i in perm:
        x = dataset[i]
        basic_model.load_state_dict(torch.load(x))
        if defense == 'AS':
            ood_choices = {
                'mnist':'emnistletters',
                'fashionmnist':'emnist',
                'cifar10':'cifar100',
                'gtsrb':'lisa',
                'imagenette':'miniimagenet'
            }
            ood_dataset = ood_choices[task]
            BATCH_SIZE, N_EPOCH, trainset, _, is_binary, _, _ = load_dataset_setting(ood_dataset)
            tot_num = len(trainset)
            shadow_ood_indices = np.random.choice(tot_num, int(tot_num * 0.02))
            print("OOD Data indices owned by the defender:", shadow_ood_indices)
            shadow_ood_set = torch.utils.data.Subset(trainset, shadow_ood_indices)
            shadow_ood_loader = torch.utils.data.DataLoader(shadow_ood_set, batch_size=BATCH_SIZE, shuffle=True)
            train_model(basic_model, shadow_ood_loader, epoch_num=N_EPOCH, is_binary=is_binary, verbose=False)
        bb.model = basic_model
        if is_discrete:
            out_benign = basic_model.emb_forward(meta_model.inp)
            out_defensed = basic_model.emb_forward(meta_model.inp, defense=defense)
        else:
            out_benign = bb(meta_model.inp, defense=None)
            out_defensed = bb(meta_model.inp, defense=defense)
        score_benign = meta_model.forward(out_benign)
        score_defensed = meta_model.forward(out_defensed)
        l = meta_model.loss(score_benign, 0)
        l += meta_model.loss(score_defensed, 1)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        cum_loss = cum_loss + l.item()
        preds.append(score_benign.item())
        preds.append(score_defensed.item())
        labs.append(0)
        labs.append(1)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.median(preds)
    acc = ((preds > threshold) == labs).mean()

    return cum_loss / (len(dataset) + len(dataset)), auc, acc


def epoch_meta_eval(bb, meta_model, basic_model, dataset, is_discrete, threshold=0.0,
                    defense=None, target=None):
    meta_model.eval()
    basic_model.train(),
    cum_loss = 0.0
    preds = []
    labs = []
    perm = np.random.permutation(len(dataset))
    for i in perm:
        x = dataset[i]
        basic_model.load_state_dict(torch.load(x))
        bb.model = basic_model
        if is_discrete:
            out_benign = basic_model.emb_forward(meta_model.inp)
            out_defensed = basic_model.emb_forward(meta_model.inp, defense=defense)
        else:
            out_benign = bb(meta_model.inp, defense=target)
            out_defensed = bb(meta_model.inp, defense=defense)
        score_benign = meta_model.forward(out_benign)
        score_defensed = meta_model.forward(out_defensed)

        preds.append(score_benign.item())
        preds.append(score_defensed.item())
        labs.append(0)
        labs.append(1)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.median(preds)
    acc = ((preds > threshold) == labs).mean()

    return cum_loss / len(preds), auc, acc


def epoch_meta_train_oc(meta_model, basic_model, optimizer, dataset, is_discrete):
    scores = []
    cum_loss = 0.0
    perm = np.random.permutation(len(dataset))
    for i in perm:
        x, y = dataset[i]
        assert y == 1
        basic_model.load_state_dict(torch.load(x))
        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)
        scores.append(score.item())

        loss = meta_model.loss(score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()
        meta_model.update_r(scores)
    return cum_loss / len(dataset)


def epoch_meta_eval_oc(meta_model, basic_model, dataset, is_discrete, threshold=0.0):
    preds = []
    labs = []
    for x, y in dataset:
        basic_model.load_state_dict(torch.load(x))
        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)

        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.asscalar(np.median(preds))
    acc = ((preds > threshold) == labs).mean()
    return auc, acc
