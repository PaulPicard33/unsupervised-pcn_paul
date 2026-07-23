import argparse
from tqdm import tqdm
from pcn import utils, plotting, datasets
import torch
from pcn.models import PCModel
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import preprocessing
from filelock import FileLock

def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    train_dataset, valid_dataset, test_dataset, size = utils.get_datasets(cf.dataset, cf.train_size, cf.test_size, cf.normalize, g)

    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size, utils.seed_worker, g)
    valid_loader = datasets.get_dataloader(valid_dataset, cf.batch_size, utils.seed_worker, g)

    if cf.dataset == 'mnist':
        cf.n_vc = 550
    elif cf.dataset == 'fmnist':
        cf.n_vc = 750
    else:
        cf.n_vc = 2000

    model_name = f"{cf.dataset}-n_vc={cf.n_vc}"
    nodes = [cf.n_vc, np.prod(size)]
    if cf.n_ec > 0:
        nodes = [cf.n_ec] + nodes
        model_name += f"-n_ec={cf.n_ec}"
    model_name += f"-seed={cf.seed}"

    model = PCModel(
        nodes=nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    model.load_state_dict(torch.load(f"models/pcn-{model_name}.pt", map_location=utils.DEVICE, weights_only=True))

    # Classification accuracy
    activities_train, labels_train = plotting.infer_latents(
        model, train_loader, cf.n_max_iters, cf.step_tolerance, cf.init_std, cf.fixed_preds_test
    )
    X_train = activities_train[cf.layer]
    y_train = labels_train
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    clf = svm.LinearSVC(verbose=1, random_state=cf.seed).fit(X_train, y_train)

    activities_valid, labels_valid = plotting.infer_latents(
        model, valid_loader, cf.n_max_iters, cf.step_tolerance, cf.init_std, cf.fixed_preds_test
    )
    X_valid = activities_valid[cf.layer]
    y_valid = labels_valid
    X_valid = scaler.transform(X_valid)
    valid_acc = clf.score(X_valid, y_valid)

    filename = f"eval_two_layers_{cf.dataset}-seed={cf.seed}.csv"
    # Lock file to prevent overwriting when multiple processes run
    with FileLock(f"{filename}.lock"): 
        print('Lock acquired.')
        with open(f'outputs/{filename}') as f:
            df = pd.read_csv(f)
            idx = df.index[df['EC size'] == cf.n_ec]
            df.loc[idx, [f'Validation accuracy {cf.layer}']] = valid_acc
            df.to_csv(f'outputs/{filename}', index=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Script that evaluates the classification accuracy of the hidden layers of a PCN."
    )
    parser.add_argument("--dataset", choices=['mnist', 'fmnist', 'cifar10'], default='mnist', help="Enter dataset name")
    parser.add_argument("--n_ec", type=int, default=30, help="Enter size of EC layer")
    parser.add_argument("--act_fn", choices=['sigmoid', 'tanh', 'relu', 'linear'], default='tanh', help="Enter activation function")
    parser.add_argument("--layer", type=int, default=1, help="Enter layer to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    # dataset params
    cf.dataset = args.dataset
    cf.train_size = None
    cf.test_size = None
    cf.label_scale = None
    cf.normalize = True
    cf.batch_size = 64

    # inference params
    cf.mu_dt = 0.01
    cf.n_max_iters = 100
    cf.step_tolerance = 1e-5
    cf.init_std = 0.01
    cf.fixed_preds_train = False
    cf.fixed_preds_test = False

    # model params
    cf.n_ec = args.n_ec
    cf.use_bias = True
    cf.kaiming_init = False
    cf.act_fn = args.act_fn
    cf.seed = args.seed
    cf.layer = args.layer

    main(cf)
