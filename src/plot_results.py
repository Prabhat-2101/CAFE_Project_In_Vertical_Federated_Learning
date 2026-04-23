import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def generate_plots():
    os.makedirs('results/plots', exist_ok=True)
    summary_path = 'results/summary.csv'
    if not os.path.exists(summary_path):
        print("No summary.csv found. Run experiments first.")
        return

    df_sum = pd.read_csv(summary_path)

    # Plot 1: Convergence Round vs Alpha (Grouped by Clients)
    plt.figure(figsize=(10, 6))
    for clients in df_sum['clients'].unique():
        subset = df_sum[df_sum['clients'] == clients]
        plt.plot(subset['alpha'].astype(str), subset['convergence_round'], marker='o', label=f"{clients} Clients")
    plt.title('Convergence Round vs Data Heterogeneity (alpha)')
    plt.xlabel('Dirichlet Alpha (IID on right)')
    plt.ylabel('Convergence Round (>80% Acc), -1 means not reached')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/convergence_vs_alpha.png')
    # plt.savefig('results/plots/convergence_vs_alpha.pdf')
    plt.close()

    # Plot 2: Final Accuracy vs Alpha
    plt.figure(figsize=(10, 6))
    for clients in df_sum['clients'].unique():
        subset = df_sum[df_sum['clients'] == clients]
        plt.plot(subset['alpha'].astype(str), subset['final_accuracy'], marker='s', label=f"{clients} Clients")
    plt.title('Final Global Test Accuracy vs Data Heterogeneity')
    plt.xlabel('Dirichlet Alpha (IID on right)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/accuracy_vs_alpha.png')
    # plt.savefig('results/plots/accuracy_vs_alpha.pdf')
    plt.close()

    # Plot 3: Communication Cost vs Clients (Bar Plot)
    plt.figure(figsize=(10, 6))
    avg_cost = df_sum.groupby('clients')['total_comm_cost_mb'].mean()
    avg_cost.plot(kind='bar', color='skyblue')
    plt.title('Average Total Communication Cost vs Client Setup')
    plt.xlabel('Number of Clients')
    plt.ylabel('Communication Cost (MB)')
    plt.grid(axis='y')
    plt.savefig('results/plots/comm_cost.png')
    # plt.savefig('results/plots/comm_cost.pdf')
    plt.close()

    # Plot 4: Accuracy Curves for all Histories
    plt.figure(figsize=(10, 6))
    history_files = glob.glob('results/history_clients*_alpha*.csv')
    for hf in history_files:
        df_hist = pd.read_csv(hf)
        name = os.path.basename(hf).replace('.csv', '').replace('history_', '')
        plt.plot(df_hist['round'], df_hist['accuracy'], label=name)
        
    plt.title('Global Test Accuracy over Rounds')
    plt.xlabel('Communication Round')
    plt.ylabel('Global Test Accuracy')
    if len(history_files) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('results/plots/accuracy_curves.png')
    # plt.savefig('results/plots/accuracy_curves.pdf')
    plt.close()

    print("All plots generated successfully in results/plots/.")

if __name__ == "__main__":
    generate_plots()
