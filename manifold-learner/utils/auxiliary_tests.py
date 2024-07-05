import torch
import numpy as np
from sklearn.utils.extmath import cartesian

class Tests:
    
    def __init__(self):
        
        self.points_dim = 150
        self.pos_dim = 6
        self.latent_max = 10
        
        self.decoded_grid = None
        self.mf = None
        self.measure = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        x = np.linspace(-self.latent_max, self.latent_max, self.points_dim)
        y = np.linspace(-self.latent_max, self.latent_max, self.points_dim)
        self.latent_grid = cartesian((x.flatten(), y.flatten()))
        
    def manifold_computation(self, model):
        
        print("Computing Manifold Visualisation...")
        
        metrics = model.embed(torch.tensor(self.latent_grid).to(self.device).float().unsqueeze_(0), jacobian=True)
        self.mf = np.array(
            [np.log(np.sqrt(np.abs(np.linalg.det(metrics[2][i])))) for i in range(self.points_dim*self.points_dim)]
        ).reshape(self.points_dim, self.points_dim)
        self.measure = np.array([
            np.sum(metrics[0][0, i , 5].cpu().detach().numpy()) for i in range(self.points_dim*self.points_dim)
        ]).reshape(self.points_dim, self.points_dim)
        self.measure = np.abs(self.measure)
        constraint = metrics[3].reshape(self.points_dim, self.points_dim)
        return self.measure, self.mf, constraint
    
    def compute_geodesic(self, model, discrete_model=None):
        """
        Future implementation of geodesic curve generation for continuous path planning. Time to whip out the stochastic approximation and RL
        """
        pass
        
        
        