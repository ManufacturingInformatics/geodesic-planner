import torch
from stochman import *
from stochman.curves import CubicSpline
import networkx as nx
import matplotlib.pyplot as plt
import copy

class DiscretizedManifold:
    
    def __init__(self, model, grid, use_diagonals=False):
        
        self.grid = grid
        self.G = nx.Graph()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if grid.shape[0] != 2:
            raise Exception('Support only for 2D grids')
        
        dim, xsize, ysize = grid.shape
        node_idx = lambda x, y: x*ysize + ysize
        self.G.add_nodes_from(range(xsize*ysize))
        
        line = CubicSpline(begin=torch.zeros(1, dim), end=torch.ones(1, dim), num_nodes=2)
        t = torch.linspace(0, 1, 5)
        self.fixed_positions = {}
        self.decoded_positions= torch.zeros((xsize*ysize, 2))
        
        for x in range(xsize):
            for y in range(ysize):
                
                line.begin = grid[:, x, y].view(1, -1)
                
                n = node_idx(x, y)
                self.fixed_positions[n] = (grid[0, x, y].detach().numpy(), grid[1, x, y].detach().numpy())
                print(model.decode(grid[:, x, y].to(device), train_rbf=False).mean.flatten().shape, self.decoded_positions.shape)
                self.decoded_positions[n] = model.decode(grid[:, x, y].to(device), train_rbf=False).mean.flatten()
                
                with torch.no_grad():
                    if x > 0:
                        line.end = grid[:, x-1, y].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x-1, y), weight=w)
                    if y > 0:
                        line.end = grid[:, x, y-1].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x, y-1), weight=w)
                    if x < xsize-1:
                        line.end = grid[:, x+1, y].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x+1, y), weight=w)
                    if y < ysize-1:
                        line.end = grid[:, x, y+1].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x, y+1), weight=w)
                    if use_diagonals and x > 0 and y > 0:
                        line.end = grid[:, x-1, y-1].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x-1, y-1), weight=w)
                    if use_diagonals and x < xsize-1 and y > 0:
                        line.end = grid[:, x+1, y-1].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x+1, y-1), weight=w)
        self.mem_g = copy.deepcopy(self.G)
        
    def draw_graph(self, graph_id, curve):
        
        plt.figure(1, figsize=(12,12))
        alpha = torch.linspace(0, 1, 500, device=curve.device).reshape((-1,1))
        latent_curves = curve(alpha).detach().numpy()
        
        pos = nx.spring_layout(self.G, fixed=self.G.nodes.keys(), pos=self.fixed_positions)
        nx.draw_networkx_nodes(self.G, pos, node_size=5)
        
        elarge = [(u, v) for (u, v, d) in self.G.edges(data=True)]
        weights = [(0, 0, 0, d['weight']) for (u, v, d) in self.G.edges(data=True)]
        
        nx.draw_networkx_edges(self.G, pos, edgelist=elarge, edge_color=weights, width=4, edge_cmap=plt.cm.blues)
        
        plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], edgecolors="yellow")
        plt.plot(latent_curves[:, 0], latent_curves[:, 1])
        plt.axis("off")
        plt.show()
        
    def grid_point(self, p):
        return (self.grid.view(self.grid.shape[0], -1) - p.view(-1,1)).pow(2).sum(dim=0).argmin().item()
    
    