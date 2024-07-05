from train import train_model, test_model
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import torch
import polars as pl

def load_data(root, num_points=None):
    
    data = pl.read_parquet(root)
    joint_positions = data.select('q0', 'q1', 'q2', 'q3', 'q4', 'q5')
    data_points = joint_positions.to_numpy()
    return data_points[:num_points,:]
    
if __name__ == "__main__":
    
    parser = ArgumentParser(
        prog="ConstraintVAE",
        description="Overarching file to train the Constraint-Aware VAE manifold"
    )   
    
    parser.add_argument('--mode', type=int)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--epochs_kl', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--rbf_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dof', type=int, default=6)
    parser.add_argument('--latent_max', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_datapoints', type=int, default=500)
    
    args = parser.parse_args()
    
    if args.num_datapoints > 3000:
        print('Number of selected points must be less than 3000. Exiting...')
        exit()
        
    print("========================================")
    # set device to cpu or cuda
    device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("========================================")
    encoder_scales = [1.0]
    
    # Creating the datasets
    print('Creating dataset...')
    data = load_data('../data/subset_data.parquet', args.num_datapoints)
    train_data = torch.utils.data.TensorDataset(
        torch.from_numpy(data).to(device).float())
    x_train, x_test = train_test_split(train_data, test_size=0.3)
    
    train_loader = torch.utils.data.DataLoader(
        x_train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=args.batch_size, shuffle=True)
    
    # Three modes:
    #   0: Training and saving only
    #   1: Testing only
    #   2: Training, saving and testing
    mode = args.mode
    
    if mode ==  0:
        model = train_model(
            args.n_samples, 
            args.lr,
            args.epochs,
            args.epochs_kl,
            args.rbf_epochs,
            train_loader,
            encoder_scales,
            args.batch_size,
            train_data,
            device)
    elif mode == 1:
        model = None
        test_model(model, train_data, train_loader, device, args.latent_max, graph_size=100)
    elif mode == 2:
        model = train_model(args.n_samples, 
            args.lr,
            args.epochs,
            args.epochs_kl,
            args.rbf_epochs,
            train_loader,
            encoder_scales,
            args.batch_size,
            train_data,
            device)
        test_model(model, train_data, train_loader, device, args.latent_max, graph_size=100)

        