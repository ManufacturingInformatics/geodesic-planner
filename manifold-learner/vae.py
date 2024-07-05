import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from sklearn.cluster import KMeans
from utils import auxiliary_tests, discretized_manifold
from utils.robot_model import Robot
from stochman.manifold import EmbeddedManifold
from stochman import nnj
import copy
from tqdm import tqdm

class VAE(nn.Module, EmbeddedManifold):
    
    def __init__(self, layers, batch_size, sigma=1e-6, sigma_z=0.1):
        
        super(VAE, self).__init__()
        self.tests = None

        self.p = int(layers[0])  # Dimension of x
        self.d = int(layers[-1])  # Dimension of z
        self.h = layers[1:-1]  # Dimension of hidden layers

        # Hyper-parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.kl_coeff = 1.0  # Automatically Set
        self.kl_coeff_max = 1.0
        self.obstacle_radius = 0.02
        self.num_clusters = 300  # Number of clusters in the RBF k_mean
        
        # Robot model for FK
        self.robot = Robot()
        
        #  Flags
        self.activate_KL = False  # if enables KL is considered in the ELBO calculation
        self.visualization = False
        self.train_var = False  # Automatically Set

        self.time_step = 0
        out_features = None
        
        enc = []
        for k in range(len(layers) - 1):
            in_features = int(layers[k])
            out_features = int(layers[k + 1])
            enc.append(nnj.BatchNorm1d(in_features))
            enc.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                         nnj.Softplus()))
        enc.append(nnj.Linear(out_features, self.d))
        
        enc_scale = []
        for k in range(len(layers) - 1):
            in_features = int(layers[k])
            out_features = int(layers[k + 1])
            enc_scale.append(nnj.BatchNorm1d(in_features))
            enc_scale.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features), nnj.Softplus()))

        dec = []
        for k in reversed(range(len(layers) - 1)):
            in_features = int(layers[k + 1])
            out_features = int(layers[k])
            dec.append(nnj.BatchNorm1d(in_features))
            dec.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                         nnj.Softplus()))
        dec.pop(0)  # remove initial batch-norm as it serves no purpose
        dec.append(nnj.Linear(out_features, self.p))

        self.encoder_loc = nnj.Sequential(*enc)
        self.decoder_loc = nnj.Sequential(*dec)
        self.encoder_scale = nnj.Sequential(*enc_scale)

        self.encoder_scale_fixed = nn.Parameter(torch.tensor([sigma_z]), requires_grad=False)
        
        self.decoder_scale = nn.Parameter(torch.tensor(sigma), requires_grad=False)
        self.dec_std = lambda z: torch.ones(20, self.p, device=self.device)

        self.prior_loc = nn.Parameter(torch.zeros(self.d), requires_grad=False)
        self.prior_scale = nn.Parameter(torch.ones(self.d), requires_grad=False)
        self.prior = td.Independent(td.Normal(loc=self.prior_loc, scale=self.prior_scale), 1)
        
    def embed(self, points, jacobian=False):

        std_scale = 1.0
        metric = None
        j = None
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)  # BxNxD
        if jacobian:
            mu_pos, j_mu = self.decode(points, train_rbf=True, jacobian=True)  # BxNxD, BxNxDx(d)
            c = self.robot.compute_constraint(mu_pos.mean.cpu().detach().numpy())
            std, j_std = self.dec_std(points, jacobian=True)  # BxNxD, BxNxDx(d)
            embedded = torch.cat((mu_pos.mean, std_scale * std),
                                 dim=2)  # BxNx(2D)
            j = torch.cat((j_mu, std_scale * j_std.squeeze(0)), dim=2)  # BxNx(2D)x(d)
            m = torch.einsum("bji,bjk->bik", j_mu, j_mu)
            m2 = torch.einsum("bji,bjk->bik", j_std.squeeze(0), j_std.squeeze(0))
            metric = (m2 + m).cpu().detach().numpy()
        else:
            mu_pos = self.decode(points, train_rbf=True, jacobian=False)  # BxNxD, BxNxDx(d)
            std = self.dec_std_pos(points, jacobian=False)  # BxNxD, BxNxDx(d)
            embedded = torch.cat((mu_pos.mean, std_scale * std),
                                 dim=2)  # BxNx(2D)
        if not is_batched:
            embedded = embedded.squeeze(0)
            if jacobian:
                j = j.squeeze(0)
        if jacobian:
            return embedded, j, metric, c
        else:
            return embedded
        
    def encode(self, x, train_rbf=False):

        z_loc = self.encoder_loc(x)
        if train_rbf:
            z_scale = self.encoder_scale(x)
        else:
            z_scale = self.encoder_scale_fixed
        z_distribution = td.Independent(td.Normal(loc=z_loc, scale=z_scale, validate_args=False), 1), z_loc
        return z_distribution

    def decode(self, z, train_rbf=False, jacobian=False, negative=False):

        ja = None
        if jacobian:
            x_loc, ja = self.decoder_loc(z.view(-1, self.d), jacobian=jacobian)
        else:
            x_loc = self.decoder_loc(z.view(-1, self.d))
        position_scale = self.decoder_scale + 1e-10

        x_var_pos = self.dec_std(z.view(-1, self.d))

        position_loc = x_loc

        x_shape = list(z.shape)
        x_shape[-1] = position_loc.shape[-1]

        position_distribution = td.Independent(
            td.Normal(loc=position_loc.view(torch.Size(x_shape)), scale=position_scale), 1)
        if train_rbf:
            # print(position_loc.shape, x_var_pos.shape, x_shape)
            position_distribution = td.Independent(
                td.Normal(loc=position_loc.view(torch.Size(x_shape)), scale=x_var_pos.view(torch.Size(x_shape))), 1)

        if jacobian:
            return position_distribution, ja
        if negative:
            return position_distribution
        return position_distribution
    
    def disable_training(self):

        for module in self.encoder_loc._modules.values():
            module.training = False
        for module in self.decoder_loc._modules.values():
            module.training = False
            
    def init_std(self, x, n_samples=20):

        self.train_var = True
        with torch.no_grad():
            _, z = self.encode(x, train_rbf=True)
        d = z.shape[1]
        inv_max_std = np.sqrt(1e-12)  # 1.0 / x.std()
        beta = 10.0 / z.std(dim=0).mean()  # 1.0
        rbf_beta = beta * torch.ones(1, self.num_clusters).to(self.device)
        k_means = KMeans(n_clusters=self.num_clusters, n_init='auto').fit(z.cpu().detach().numpy())
        centers = torch.tensor(k_means.cluster_centers_)
        self.dec_std = nnj.Sequential(nnj.RBF(d, self.num_clusters, points=centers, beta=rbf_beta),
                                          # d --> num_clusters
                                          nnj.PosLinear(self.num_clusters, 1, bias=False),  # num_clusters --> 1
                                          nnj.Reciprocal(inv_max_std),  # 1 --> 1
                                          nnj.PosLinear(1, 6))  # 1 --> D
        self.dec_std.to(self.device)
        cluster_centers = k_means.cluster_centers_
        return cluster_centers
    
    def fit_std(self, data_loader, num_epochs, model, n_samples):

        params = list(self.encoder_scale.parameters()) + list(self.dec_std.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4)
        td = tqdm(range(num_epochs), colour='green')
        for epoch in td:
            for batch_idx, (data,) in enumerate(data_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                loss, loss_kl, loss_log = loss_function_elbo(data, model, train_rbf=True, n_samples=n_samples)
                loss.backward()
            optimizer.step()
            # print('Training RBF Networks ====> Epoch: {}/{} | ELBO: {}'.format(epoch+1, num_epochs, loss))
            td.set_description(f'ELBO: {loss}')
            
def loss_function_elbo(x, model, train_rbf, n_samples):

    q, _ = model.encode(x, train_rbf=train_rbf)

    z = q.rsample(torch.Size([n_samples]))  # (n_samples)x(batch size)x(latent dim)
    px_z = model.decode(z, train_rbf=train_rbf, negative=True)  # p(x|z)

    log_p_negative = log_prob(model, x, px_z)  # vMF(x|mu(z), k(z))
    log_p_negative = torch.mean(log_p_negative) * 1

    log_p = log_p_negative
    kl = torch.tensor([0.0])
    if model.activate_KL:
        log_p = log_p_negative
        kl = -0.5 * torch.sum(1 + q.variance.log() - q.mean.pow(2) - q.variance) * model.kl_coeff
        kl = kl * 400000000
        elbo = torch.mean(log_p - kl, dim=0)
    else:
        elbo = torch.mean(log_p, dim=0)
    log_mean = torch.mean(log_p, dim=0)
    return -elbo, kl, log_mean
    
def log_prob(model, x, positional_dist):

    log_p = torch.mean(positional_dist.log_prob(x), dim=1)
    log_likelihood = log_p
    return log_likelihood