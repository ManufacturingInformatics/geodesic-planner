import torch
from vae import VAE, loss_function_elbo
from utils.discretized_manifold import DiscretizedManifold
from utils.auxiliary_tests import Tests
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import scienceplots
import pandas as pd
from datetime import datetime
import pickle

NUM_REPITITIONS = 5

plt.style.use(['science', 'nature'])

def train_model(n_samples, learning_rate, epochs, epochs_kl, epochs_rbf, train_loader, encoder_scale=[1.0], batch_size=64, train_data=None, device='cpu'):

    losses = np.zeros((epochs+epochs_kl, NUM_REPITITIONS))
    schema = []
    
    models = {}
    
    for rep in range(NUM_REPITITIONS):
        print(f'\nRepitition: {rep+1}/{NUM_REPITITIONS}')
        
        model = VAE(layers=[6, 200, 100, 2], batch_size=batch_size, sigma_z=encoder_scale).to(device)
        
        print("Focusing on regularization...")

        # regularization focused training
        model.activate_KL = True
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = lambda data, train_rbf: loss_function_elbo(data, model, train_rbf, n_samples=n_samples)

        td_kl = tqdm(range(int(epochs_kl)), colour='green')
        for epoch in td_kl:
            loss = train(model, optimizer, loss_function, train_loader, epoch, device, batch_size, train_rbf=False)
            losses[epoch, rep] = loss
            td_kl.set_description(f'ELBO: {loss}')
            
        print("Focusing on reconstruction...")

        # Focusing on reconstruction of the data
        model.activate_KL = False
        model.kl_coeff = 0.1
        params = list(model.decoder_loc.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        td = tqdm(range(int(epochs)), colour='green')
        for epoch in td:
            loss = train(model, optimizer, loss_function, train_loader, epoch, device, batch_size, train_rbf=False)
            losses[epoch+epochs_kl, rep] = loss
            td.set_description(f'ELBO: {loss}')
        
        # Train RBF/Variance networks
        if train_data is None:
            print("No training data provided. Skipping...")
        else:
            print("Training RBF Variance networks...")
            model.init_std(train_data.tensors[0].float())
            model.fit_std(train_loader, epochs_rbf, model, n_samples=n_samples)
            
        models[f'rep_{rep}'] = {'model': model, 'losses': losses[:, rep]}
        schema.append(f'rep{rep}')
        
        # Saving the model into a file
        fn = './model/vae_loss_{}_rep_{}_{}.pt'.format(int(loss), rep, datetime.now().date())
        print(f'Saving model: {fn}')
        torch.save({'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'repetition': rep,
                    'encoder_scale': encoder_scale}, fn)
        
        pickle.dump(train_data, open('./model/train_data_rep_{}.p'.format(rep), 'wb'))
        
    df = pd.DataFrame(losses, columns=schema)
    df['mean'] = df.mean(axis=1)
    df['std'] = df[schema].std(axis=1)

    plt.figure(figsize=(4,4))
    plt.plot(df['mean'], 'b', label='Mean ELBO Loss')
    plt.fill_between(df.index, df['mean']-df['std'], df['mean']+df['std'], alpha=0.5, facecolor='b', label='$\sigma$-Variation')
    plt.xlim([0, len(losses)])
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('ELBO Loss', fontsize=12)
    plt.legend(loc='upper right')
    # plt.savefig('./elbo_loss.png')
    # plt.savefig('./elbo_loss.eps')
    plt.show()
    
    return model

def train(model, optimizer, loss_function, data_loader, epoch, device, batch_size, train_rbf):
    
    model.train() # Sets the model into training mode as opposed to evaluation mode
    for batch_idx, (data,) in enumerate(data_loader):
        data = data.to(device)
        # prevent crashing when the leftover training data is not enough got an epoch
        if data.shape[0] != batch_size:
            break
        optimizer.zero_grad()
        batch_loss, loss_kl, loss_log = loss_function(data, train_rbf)
        batch_loss.backward()
        optimizer.step()
    return batch_loss

def test_model(model, train_data, train_loader, device, latent_max, graph_size):
    
    model.init_std(train_data.tensors[0].float())
    encoder_scale = model.encoder_scale
    model.to(device)
    model.disable_training()
    
    ran = torch.linspace(-latent_max, latent_max, graph_size)
    x, y = torch.meshgrid(ran, ran, indexing='ij')
    grid = torch.cat((x.unsqueeze(0), y.unsqueeze(0)))
    print("Computing manifold...")
    # discrete_model = DiscretizedManifold(model, grid)
    tests = Tests()
    var_measure, mag_fac, constraint = tests.manifold_computation(model)
    
    plot(model, var_measure, mag_fac, constraint, device, train_loader, latent_max=latent_max)
    
def plot(model, var_measure, mag_fac, constraint, device, train_loader=None, latent_max=10):
    
    if train_loader is None:
        
        fig, ax = plt.subplots(figsize=(13,6))
        ax.axis('off')
    
        ax1 = plt.subplot(121)
        val = ax1.imshow(np.rot90(var_measure), interpolation='bicubic', extent=[-latent_max, latent_max, -latent_max, latent_max])
        fig.colorbar(val, ax=ax1, location='right', anchor=(0, 0.3), shrink=0.7)
        ax1.grid(False)
        # ax1.legend(fontsize=18, loc='upper right')
        ax1.set_xlim(-latent_max, latent_max)
        ax1.set_ylim(-latent_max, latent_max)
        
        ax2 = fig.add_subplot(122)
        val2 = ax2.imshow(np.rot90(mag_fac), interpolation='bicubic',
                extent=[-latent_max, latent_max, -latent_max, latent_max])
        ax2.set_xlim(-latent_max, latent_max)
        ax2.set_ylim(-latent_max, latent_max)
        fig.colorbar(val2, ax=ax2, location='right', anchor=(0, 0.3), shrink=0.7)
        # fig.tight_layout()
        ax2.grid(False) 
        
    else:
        
        # Reconstruct the data from the dataset
        model.eval()
        data_points_reconstructed = np.zeros((len(train_loader.dataset), 2))
        td = tqdm(range(len(train_loader.dataset[:])), desc='Encoding...', colour='red')
        for data_index in td:
            data_points = train_loader.dataset[:][data_index][0].cpu().detach().numpy()
            encoded = model.encode(torch.tensor(data_points).to(device), train_rbf=True)
            data_points_reconstructed[data_index] = encoded[0].mean.cpu().detach().numpy()
            
        
        fig, ax = plt.subplots(figsize=(4,4))
        val = ax.imshow(var_measure, interpolation='bicubic', extent=[-latent_max, latent_max, -latent_max, latent_max])
        fig.colorbar(val, ax=ax, location='right', fraction=0.046, pad=0.04)
        ax.set_xlabel('Latent Dim 1', fontsize=12)
        ax.set_ylabel('Latent Dim 2', fontsize=12)
        ax.grid(False)
        # fig.savefig('./var_measure.eps')
        # fig.savefig('./var_measure.png')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(4,4))
        val2 = ax.imshow(np.rot90(mag_fac), interpolation='bicubic', extent=[-latent_max, latent_max, -latent_max, latent_max])
        fig.colorbar(val2, ax=ax, location='right', fraction=0.046, pad=0.04)
        ax.set_xlabel('Latent Dim 1', fontsize=12)
        ax.set_ylabel('Latent Dim 2', fontsize=12)
        ax.set_xlim(-latent_max, latent_max)
        ax.set_ylim(-latent_max, latent_max)
        ax.grid(False)
        ax.scatter(data_points_reconstructed[:, 0], data_points_reconstructed[:, 1], marker='.', color='white', s=50, label='Training data', alpha=0.25)
        # fig.savefig('./mag_fac.eps')
        # fig.savefig('./mag_fac.png')
        plt.show()
        
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(-latent_max, latent_max, 100)
        y = np.linspace(-latent_max, latent_max, 100)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, var_measure, facecolors=cm.jet(constraint/np.amax(constraint)))
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(constraint/np.amax(constraint))
        fig.colorbar(m, ax=ax, location='left', fraction=0.046, pad=0.005, shrink=0.4)
        ax.set_xlabel('Latent Dim 1', fontsize=10)
        ax.set_ylabel('Latent Dim 2', fontsize=10)
        ax.set_zlabel('Variance Measure', fontsize=10)
        ax.set_box_aspect(aspect=None, zoom=0.8)
        plt.tight_layout()
        # plt.savefig('./metric.pdf')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(4,4))
        ax.axis('off')
        
        ax1 = plt.subplot(121)
        val = ax1.imshow(var_measure, interpolation='bicubic', extent=[-latent_max, latent_max, -latent_max, latent_max])
        fig.colorbar(val, ax=ax1, location='top', fraction=0.046, pad=0.04)
        
        path = pd.read_csv('../data/paths/on-manifold_1.csv').to_numpy(dtype=np.double)
        path_encoded = np.zeros((path.shape[0], 2))
        td = tqdm(range(path.shape[0]), desc='Embedding trajectories..')
        for i in td:
            data_points = path[i, :]
            encoded = model.encode(torch.tensor(data_points).to(device).float(), train_rbf=True)
            path_encoded[i] = encoded[0].mean.cpu().detach().numpy()
        
        ax1.plot(path_encoded[:,0], path_encoded[:,1], color='white', marker='.', linewidth=2, markersize=3)
    
        ax1.set_xlabel('Latent Dim 1', fontsize=12)
        ax1.set_ylabel('Latent Dim 2', fontsize=12)
        ax1.set_xlim(-latent_max, latent_max)
        ax1.set_ylim(-latent_max, latent_max)
        ax1.grid(False)
        
        ax2 = plt.subplot(122)
        ax2.imshow(var_measure, interpolation='bicubic', extent=[-latent_max, latent_max, -latent_max, latent_max])
        path = pd.read_csv('../data/paths/on-manifold_1.csv').to_numpy(dtype=np.double)
        path_encoded = np.zeros((path.shape[0], 2))
        td = tqdm(range(path.shape[0]), desc='Embedding trajectories..')
        for i in td:
            data_points = path[i, :]
            encoded = model.encode(torch.tensor(data_points).to(device).float(), train_rbf=True)
            path_encoded[i] = encoded[0].mean.cpu().detach().numpy()
        
        ax2.plot(path_encoded[:,0], path_encoded[:,1], color='white', marker='.', linewidth=2, markersize=3)
        
        xmax = 0
        xmin = 0
        ymax = 0
        ymin = 0
        
        for i in range(3):
        
            path = pd.read_csv(f'../data/paths/off-manifold_{i+1}.csv').to_numpy(dtype=np.double)
            path_encoded = np.zeros((path.shape[0], 2))
            td = tqdm(range(path.shape[0]), desc='Embedding trajectories..')
            for i in td:
                data_points = path[i, :]
                encoded = model.encode(torch.tensor(data_points).to(device).float(), train_rbf=True)
                path_encoded[i] = encoded[0].mean.cpu().detach().numpy()
            
            ax2.plot(path_encoded[:,0], path_encoded[:,1], color='red', marker='.', linewidth=0.5, markersize=1)
            
            if np.amin(path_encoded[:,0]) < xmin:
                xmin = np.amin(path_encoded[:,0])
            if np.amax(path_encoded[:,0]) > xmax:
                xmax = np.amax(path_encoded[:,0])
            if np.amin(path_encoded[:,1]) < ymin:
                ymin = np.amin(path_encoded[:,1])
            if np.amax(path_encoded[:,1]) > ymax:
                ymax = np.amax(path_encoded[:,1])
        
        # ax2.set_xlabel('Latent Dim 1', fontsize=12)
        # ax2.set_ylabel('Latent Dim 2', fontsize=12)
        ax2.set_xlim(xmin-0.1, xmax+0.1)
        ax2.set_ylim(ymin-0.1, ymax+0.1)
        asp = np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0]
        ax2.set_aspect(asp)
        ax2.grid(False)
        plt.show()
    
