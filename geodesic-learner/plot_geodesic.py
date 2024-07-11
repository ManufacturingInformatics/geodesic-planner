import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from environment import CurveEnv
import scienceplots
import torch
import argparse

plt.style.use(['science', 'ieee'])
plt.rcParams.update({'figure.dpi': '100'})

def main() -> None:
    
    parser = argparse.ArgumentParser(
        description='Plotting Geodesic Policy Search via PPO'
    )
    parser.add_argument('-r', '--rep', type=int)
    
    args = parser.parse_args()
    
    df = pl.read_parquet('./geodesic_model/policy_params_{}.parquet'.format(args.rep))
    
    var_measure_const = np.loadtxt('./data/heat_map.csv')
    device = torch.device('cpu')

    env = CurveEnv((1,7), (8,4.5), device, np.rot90(var_measure_const))
    
    size = 20
    latent_max = 10

    t = np.linspace(0, 1, size)
    # x = np.zeros((size,))
    # y = np.zeros((size,))

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    arr = df[-1].to_numpy().reshape((8,)).tolist()
    [ax, bx, cx, dx, ay, by, cy, dy] = arr

    x = ax*t**3 + bx*t**2 + cx*t + dx
    y = (ay*t**3 + by*t**2 + cy*t + dy)+6

    ax1 = plt.subplot(111)
        
    ax1.set_xlabel('$\mathbf{z}_1$', fontsize=15)
    ax1.set_ylabel('$\mathbf{z}_2$', fontsize=15)
    val = ax1.imshow(env.M, extent=[-latent_max, latent_max, -latent_max, latent_max])
    fig.colorbar(val, ax=ax1, location='top', fraction=0.046, pad=0.04)

    ax2 = ax1.twinx().twiny()
    ax2.plot(x, y, '#E87313', zorder=2, linestyle='-')
    ax2.scatter(env.range[0,0], env.range[0,1], c='#FF0000', s=10, zorder=2)
    ax2.scatter(env.range[1,0], env.range[1,1], c='#19FF00', s=10, zorder=2)

    ax2.set_xlim([-1, 11])
    ax2.set_ylim([-1, 11])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    
    plt.show()

if __name__ == "__main__":
    
    main()