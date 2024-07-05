import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import sympy as spy
import scipy
import math as m
import scipy as sp
import torch
import torchquad
from torchquad import Simpson

class CurveEnv(gym.Env):
    
    def __init__(self, start_node, end_node, device, heat_map, num_params=4):
        
        super().__init__()
        
        self.start = start_node
        self.end = end_node
        self.range = np.array([self.start, self.end])
        self.num_params = num_params
        
        self.sigma = 3
        self.A = 250
        
        self.K_d = 0.1
        self.K_l = 10
        
        self._max_episode_steps = 200
        
        # x = np.linspace(-1, 11, 150)
        # y = np.linspace(-1, 11, 150)
        # self.X, self.Y = np.meshgrid(x, y)
        # self.M = 1+self.A*np.exp(-((self.X-5)**2)/(2*self.sigma**2) - ((self.Y-5)**2)/(2*self.sigma**2))
        ran = np.linspace(-1, 11, 150)
        self.X, self.Y = np.meshgrid(ran, ran)
        
        self.M = heat_map
        self.metric = np.array([self.X.flatten(), self.Y.flatten(), self.M.flatten()]).T
        
        # Storing the CUDA device here to ensure that all tensors are in the same place
        self.device = device
        
        # Gompertz function parameters. The value of a is static to allow a range [-1,1], but the values of b and c can be tuned hyperparameters
        self.a_d = 2
        self.b_d = 0.61595
        self.c_d = 2.5
        
        self.b_l = 5.844
        self.c_l = 4.854
        
        self.size = 20
        
        # Numerical integration calculator for curve length
        self.simp = Simpson()
        self.N = 101
        
        # Setting the beginning parameters of the curve
        self.dx = self.start[0]
        self.dy = self.start[1]
        
        # Inheriting some stuff from gym.Env
        self.observation_space = gym.spaces.Box(low=-np.iinfo(np.int64).max+1, high=np.iinfo(np.int64).max-1, shape=(self.num_params,))
        self.action_space = gym.spaces.Box(low=-5, high=5, shape=(self.num_params,), dtype=np.float32)
        self.t = np.linspace(0, 1, self.size)
    
        self.visualisation = GraphVisualiser(self.range, self.M)
        self.state_old, self.cl_old, _ = self.reset()
    
    def step(self, step, action):
        
        state = self.state_old + action
        x, y, params = self._get_curve(state)
        cl = self._get_curve_length(params)
        delta = self.cl_old - cl
        reward = self.reward(cl, delta)
        
        if step == self._max_episode_steps:
            done = True
        else:
            done = False
        
        return state, reward, [x,y], done, params, cl # State in this case is the curve length
    
    def reset(self):
        
        info = {}
        action = self.action_space.sample()
        state = np.zeros((self.num_params,)) + action
        x, y, params = self._get_curve(state)
        L = self._get_curve_length(params)
        self.render(0, x, y, L)
        return state, L, info
    
    def _get_curve(self, action):
        
        [ax, ay, cx, cy] = action
        bx = self.end[0] - ax - cx - self.dx
        by = self.end[1] - ay - cy - self.dy
        
        x = ax*self.t**3 + bx*self.t**2 + cx*self.t + self.dx
        y = ay*self.t**3 + by*self.t**2 + cy*self.t + self.dy
        
        curve_params = [ax, bx, cx, ay, by, cy]
        
        return x, y, curve_params
    
    def _get_curve_length(self, params):
        
        ax, bx, cx, ay, by, cy = params
        
        def metric(x, y):
            # print(x, y)
            xy = np.array([x.detach().cpu().numpy(), y.detach().cpu().numpy()]).T
            metric = self.metric[np.all(np.isclose(self.metric[:, :2], xy, rtol=1, atol=1), axis=1), -1]
            return metric[0]
        
        def func(t):
            x = (ax*3)*t**2 + (bx*2)*t + (cx)
            y = (ay*3)*t**2 + (by*2)*t + (cy)
            M = torch.zeros(101,1)
            for i in range(x.shape[0]):
                temp = metric(x[i], y[i])
                if temp == torch.inf:
                    M[i] = torch.tensor([1e+6])
                else:
                    M[i] = torch.tensor([temp])
            # print(M, x, y)
            return torch.sqrt(M*(x**2 + y**2))
        
        curve_length = self.simp.integrate(
            func,
            dim=1,
            N=self.N,
            integration_domain=[[0,1]],
            backend='torch'
        )
        return curve_length
    
    def reward(self, state, delta):
        
        state_normalise = self._scaler_length(state)
        delta_normalise = self._scaler_delta(delta)
        
        if delta_normalise < 0:
            r_d = 0
        else:
            r_d = -1 + self.a_d*torch.exp(-self.b_d*torch.exp(-self.c_d*delta_normalise))
            
        r_l = torch.exp(-self.b_l*torch.exp(-self.c_l*state_normalise))

        return self.K_l*r_l + self.K_d*r_d
    
    def render(self, step, x, y, length, mode='live'):
            
        self.visualisation._render_curve(x, y, step, length)
        
    def _scaler_length(self, state):
        return torch.Tensor([state/(1+state)]).to(self.device)
        
    def _scaler_delta(self, delta):
        return torch.Tensor([2/(1+torch.exp(torch.Tensor([-delta]).to(self.device)))-1]).to(self.device)
    
class GraphVisualiser:
    
    def __init__(self, range, M):
        
        plt.ion()
        
        self.fig = plt.figure(figsize=(5, 5)) # figsize=(3.5, 3.5)
        
        self.ax = plt.subplot(111)
        self.range = range
        self.M = M
        
        self._set_params()
        # plt.show(block=False)
        
    def _render_curve(self, x, y, step=None, cl=0):
        
        if step is None:
            self.fig.suptitle('Spline Visualiser')
        else:
            self.fig.suptitle(f'Iteration: {step+1} | Curve Length: {cl}')
        
        # self.ax.cla()
        
        self.ax.plot(x, y, '#B0ACAB', zorder=1, linestyle='-')
        self.fig.canvas.flush_events()
        
    def _reset_graph(self, x, y):
        
        self.ax.cla()
        self._set_params()
        self.ax.plot(x, y, '#B0ACAB', zorder=1, linestyle='-')
        # self.ax.grid()
        
        
    def _set_params(self):
        
        self.ax.set_xlabel('$\mathbf{x}$')
        self.ax.set_ylabel('$\mathbf{y}$')
        self.ax.set_xlim([-1, 11])
        self.ax.set_ylim([-1, 11])
        
        self.ax.imshow(self.M, extent=[-1, 11, -1, 11])
        
        self.ax.scatter(self.range[0,0], self.range[0,1], c='#FF0000', s=10, zorder=2)
        self.ax.scatter(self.range[1,0], self.range[1,1], c='#19FF00', s=10, zorder=2)
        # self.ax.grid()
        
    def close(self):
        
        plt.close()
        