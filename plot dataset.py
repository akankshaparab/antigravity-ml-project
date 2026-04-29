import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a premium, modern style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial', 'DejaVu Sans']

def plot_class_distribution():
    # Load the dataset labels
    file_path = 'spider_final_embeddings.npz'
    
    try:
        data = np.load(file_path, allow_pickle=True)
        y = data['y']
    except FileNotFoundError:
        # Fallback to reduced embeddings if final doesn't exist yet
        file_path = 'spider_reduced_embeddings.npz'
        data = np.load(file_path, allow_pickle=True)
        y = data['y']

    # Count the occurrences of each class
    unique_labels, counts = np.unique(y, return_counts=True)
    
    # Sort them in a logical difficulty progression
    order = ["Easy", "Medium", "Hard", "Extra Hard"]
    order = [label for label in order if label in unique_labels]
    
    # Reorder counts to match the sequence
    label_to_count = dict(zip(unique_labels, counts))
    ordered_counts = [label_to_count[label] for label in order]

    # Calculate percentages for labels
    total = sum(counts)
    percentages = [(count / total) * 100 for count in ordered_counts]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Use a vibrant, sophisticated palette
    colors = sns.color_palette("viridis", n_colors=len(order))
    
    # Create the bars
    bars = ax.bar(order, ordered_counts, color=colors, edgecolor='#ffffff', linewidth=2, width=0.6)
    
    # Add numerical labels and percentages above each bar
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height + (total * 0.01), 
            f'{height:,}\n({pct:.1f}%)', 
            ha='center', 
            va='bottom', 
            fontweight='bold', 
            fontsize=12,
            color='#2c3e50'
        )

    # Clean up spines and configure grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.grid(False)
    
    # Titles and Axis formatting
    ax.set_title('Spider Dataset: SQL Complexity Distribution', fontsize=20, fontweight='bold', pad=25, color='#1a1a1a')
    ax.set_xlabel('Difficulty Level', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Number of Queries', fontsize=14, fontweight='bold', labelpad=15)
    
    plt.yticks(color='#666666')
    plt.xticks(fontweight='bold', color='#2c3e50')
    
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#ffffff')

    plt.tight_layout()
    
    # Export and display
    output_filename = 'spider_class_distribution.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Chart saved as '{output_filename}'")
    plt.show()

if __name__ == '__main__':
    plot_class_distribution()
