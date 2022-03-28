import torch.utils.data
import argparse
from tqdm import tqdm
from utils_meta import load_model_setting, epoch_meta_train, epoch_meta_eval
from offline.meta_classifier import MetaClassifier
from model_lib.defense_device import Victim

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/gtsrb/fashionmnist/imagenette).')
parser.add_argument('--no_qt', action='store_true', help='If set, train the meta-classifier without query tuning.')
parser.add_argument('--defense', type=str, required=True, help='Specify which defense is used('
                                                               'ReverseSigmoid/ClassLabels/HighConfidence'
                                                               '/GaussianNoise/Rounding/MAD).')
parser.add_argument('--target', type=str, required=False, default=None, help='Defense deployed in target model: {None/MAD/ReverseSigmoid/...}')
parser.add_argument('-d', default=-1, help='-1 ~ cpu, n > 0: nth gpu will be assigned')

if __name__ == '__main__':
    args = parser.parse_args()

    GPU = True
    N_REPEAT = 5
    N_EPOCH = 10
    TRAIN_NUM = 2048
    # TO DO : 2048
    VAL_NUM = 256
    # To DO : 256
    TEST_NUM = 256
    # TO DO : 256

    if args.no_qt:
        save_path = './meta/%s_no-qt.model' % args.task
    else:
        save_path = './meta/%s' % args.defense + '/%s.model' % args.task
    shadow_path = './shadow/%s/models' % args.task

    Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting(args.task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if inp_mean is not None:
        inp_mean = torch.FloatTensor(inp_mean)
        inp_std = torch.FloatTensor(inp_std)
        if GPU:
            inp_mean = inp_mean.to(device)
            inp_std = inp_std.to(device)
    print("Task: %s; input size: %s; class num: %s" % (
        args.task, input_size, class_num))

    train_dataset = []
    for i in range(TRAIN_NUM):
        x = shadow_path + '/shadow_%d.model' % i
        train_dataset.append(x)

    val_dataset = []
    for i in range(TRAIN_NUM, TRAIN_NUM + VAL_NUM):
        x = shadow_path + '/shadow_%d.model' % i
        val_dataset.append(x)

    test_dataset = []
    for i in range(TEST_NUM):
        x = shadow_path + '/target_%d.model' % i
        test_dataset.append(x)

    bb = Victim(gpu=GPU)
    AUCs = []
    for i in range(N_REPEAT):
        shadow_model = Model(gpu=GPU)
        target_model = Model(gpu=GPU)
        meta_model = MetaClassifier(input_size, class_num, N_in=1, gpu=GPU)
        if inp_mean is not None:
            init_inp = torch.zeros_like(meta_model.inp).normal_() * inp_std + inp_mean
            meta_model.inp.data = init_inp.to(device)
        else:
            meta_model.inp.data = meta_model.inp.data

        if not args.load_exist:
            print("Training Meta Classifier %d/%d" % (i + 1, N_REPEAT))
            if args.no_qt:
                print("No query tuning.")
                optimizer = torch.optim.Adam(list(meta_model.fc.parameters()) + list(meta_model.output.parameters()),
                                             lr=1e-3)
            else:
                optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)

            best_eval_auc = None
            test_info = None
            for _ in tqdm(range(N_EPOCH)):
                epoch_meta_train(bb, meta_model, shadow_model, optimizer, train_dataset,
                                 is_discrete=is_discrete,
                                 # threshold='half',
                                 defense=args.defense, task=args.task)
                eval_loss, eval_auc, eval_acc = epoch_meta_eval(bb, meta_model, shadow_model, val_dataset,
                                                                is_discrete=is_discrete,
                                                                # threshold='half',
                                                                defense=args.defense)
                if best_eval_auc is None or eval_auc > best_eval_auc:
                    best_eval_auc = eval_auc
                    test_info = epoch_meta_eval(bb, meta_model, shadow_model, test_dataset, is_discrete=is_discrete,
                                                # threshold='half',
                                                defense=args.defense)
                    torch.save(meta_model.state_dict(), save_path + '_%d' % i)
        else:
            print("Evaluating Meta Classifier %d/%d" % (i + 1, N_REPEAT))
            meta_model.load_state_dict(torch.load(save_path + '_%d' % i))
            test_info = epoch_meta_eval(bb, meta_model, target_model, test_dataset, is_discrete=is_discrete,
                                        # threshold='half',
                                        defense=args.defense, target=args.target)

        print("\tTest AUC:", test_info[1])
        AUCs.append(test_info[1])

    AUC_mean = sum(AUCs) / len(AUCs)
    torch.save(AUC_mean, save_path + '.auc_mean.pth')

    print("Average detection AUC on %d meta classifier: %.4f" % (N_REPEAT, AUC_mean))
