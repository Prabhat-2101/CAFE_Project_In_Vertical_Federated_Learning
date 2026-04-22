import os
import yaml
import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import flwr as fl
from src.data import load_data
from src.client import FlowerClient
from src.server import get_evaluate_fn
from src.model import CNN

def get_model_size_mb():
    model = CNN()
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    with open('configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config['seed'])
    os.makedirs('results', exist_ok=True)
    
    clients_list = config['experiments']['clients']
    alphas_list = config['experiments']['alphas']
    num_rounds = 20 # Let's set some default max rounds if we want,
    summary_data = []

    model_size_mb = get_model_size_mb()
    
    for num_clients in clients_list:
        for alpha in alphas_list:
            print(f"\\n--- Running experiment with {num_clients} clients, alpha={alpha} ---")
            
            # 1. Load Data
            client_trainloaders, testloader = load_data(len(client_trainloaders) if 'client_trainloaders' in locals() else num_clients, alpha, config['batch_size']) # Actually, length is always num_clients.
            client_trainloaders, testloader = load_data(num_clients, alpha, config['batch_size'])
            
            # 2. Define Client Factory
            def client_fn(cid: str) -> fl.client.Client:
                loader = client_trainloaders[int(cid)]
                return FlowerClient(loader, testloader, config['local_epochs'], config['lr'], config['momentum']).to_client()

            # 3. Strategy
            eval_fn = get_evaluate_fn(testloader)
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=config['fraction_fit'],
                fraction_evaluate=0.0, # we do central evaluation
                min_available_clients=num_clients,
                evaluate_fn=eval_fn
            )
            
            # 4. Start Simulation
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=num_clients,
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy,
            )
            
            # 5. Extract Metrics
            losses_distributed = history.losses_distributed
            losses_centralized = history.losses_centralized
            metrics_centralized = history.metrics_centralized
            
            accuracies = [m[1] for m in metrics_centralized['accuracy']]
            
            # Convergence round (>80% accuracy)
            convergence_round = -1
            for r, acc in enumerate(accuracies):
                if acc > 0.8:
                    convergence_round = r + 1 # 1-indexed based rounds
                    break
                    
            # Communication Cost (MB)
            # Each participating client sends params to server (1x) and receives from server (1x). 
            participating_per_round = int(num_clients * config['fraction_fit'])
            total_comm_cost = model_size_mb * 2 * participating_per_round * num_rounds
            
            # Save historical data
            df_hist = pd.DataFrame({
                'round': [i+1 for i in range(num_rounds)],
                'loss': [l[1] for l in losses_centralized] if losses_centralized else [0]*num_rounds,
                'accuracy': accuracies if accuracies else [0]*num_rounds
            })
            df_hist.to_csv(f"results/history_clients{num_clients}_alpha{alpha}.csv", index=False)
            
            summary_data.append({
                'clients': num_clients,
                'alpha': alpha,
                'convergence_round': convergence_round,
                'total_comm_cost_mb': total_comm_cost,
                'final_accuracy': accuracies[-1] if accuracies else 0
            })

    # Save summary
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv('results/summary.csv', index=False)
    print("Experiments Finished. Summary saved directly to results/summary.csv.")
    
if __name__ == "__main__":
    main()
