import argparse
from pcn import utils, plotting, datasets
import torch
from pcn.models import PCModel
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

def main(cf):

    utils.seed(cf.seed)
    g = torch.Generator()
    g.manual_seed(cf.seed)

    train_dataset, valid_dataset, test_dataset, size = utils.get_datasets(
        cf.dataset, 
        cf.train_size, 
        cf.test_size, 
        cf.normalize, 
        g)
    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size, utils.seed_worker, g)
    test_loader = datasets.get_dataloader(test_dataset, cf.batch_size, utils.seed_worker, g)

    if cf.dataset == 'mnist':
        cf.n_vc = 450
        cf.n_ec = 30
    elif cf.dataset == 'fmnist':
        cf.n_vc = 750
        cf.n_ec = 300
    else:
        cf.n_vc = 2000
        cf.n_ec = 300

    cf.nodes = [cf.n_ec, cf.n_vc, np.prod(size)]

    model_name = f"pcn-{cf.dataset}-n_vc={cf.n_vc}-n_ec={cf.n_ec}"

    model = PCModel(
        nodes=cf.nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=utils.DEVICE, weights_only=True))

    activities_train, labels_train = plotting.infer_latents(
        model, train_loader, cf.n_max_iters, cf.step_tolerance, cf.init_std, cf.fixed_preds_test
    )

    # From fitted Euclidean Gaussian
    classes = np.unique(train_dataset.dataset.targets)
    labels = []
    ec_batch_euclid = []
    ec_stat = {'mu': [], 'cov': []}
    for i in classes:
        indices = np.where(np.array(labels_train) == i)[0]
        z0_train = np.array(activities_train[0])[indices]
        z0 = utils.sample_from_latent(z0_train, cf.batch_size)
        ec_batch_euclid.append(utils.set_tensor(z0))
        labels += [i for _ in range(cf.batch_size)]

    activities_test, labels_test = plotting.infer_latents(
        model, test_loader, cf.n_max_iters, cf.step_tolerance, cf.init_std, cf.fixed_preds_test
    )
    fig1, fig2 = plotting.visualize_samples(
        model, 
        cf, 
        activities_test, 
        labels_test, 
        ec_batch_euclid, 
        labels, 
        size, 
        cf.horizontal)
    filename = 'gen-replay'
    if cf.horizontal:
        filename += '-horizontal'
    fig1.savefig(f"outputs/{model_name}/{filename}.png")
    fig2.savefig(f'outputs/{model_name}/latents-{filename}.png')
    torch.save(ec_stat, f"outputs/{model_name}/ec_stat.pt")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Script that evaluates the generative replay of a PC model"
    )
    parser.add_argument("--dataset", choices=['mnist', 'fmnist', 'cifar10'], default='mnist', help="Enter dataset name")
    parser.add_argument("--horizontal", action='store_true', help="Enable horizontal figure format")
    parser.add_argument("--seed", type=int, default=0, help="Enter seed")
    args = parser.parse_args()

    # Hyperparameters dict
    cf = utils.AttrDict()

    # experiment params
    cf.seed = args.seed

    # dataset params
    cf.dataset = args.dataset
    cf.train_size = None
    cf.test_size = None
    cf.label_scale = None
    cf.normalize = False if cf.dataset == "mnist" else True
    cf.batch_size = 64

    # inference params
    cf.mu_dt = 0.01
    cf.n_train_iters = 50
    cf.n_test_iters = 200
    cf.n_max_iters = 10000
    cf.step_tolerance = 1e-5
    cf.init_std = 0.01
    cf.fixed_preds_train = False
    cf.fixed_preds_test = False    

    # model params
    cf.use_bias = True
    cf.kaiming_init = False
    cf.act_fn = "tanh"

    cf.horizontal = args.horizontal

    main(cf)