import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_comparison_table():
    # Data for the 4 variants
    variants = ["Standard PCA", "Incremental PCA", "Kernel PCA", "Sparse PCA"]
    metrics = ["Time (s)", "MSE", "Variance Explained (%)"]
    
    # Example values (you can replace these with actual results from phase 2.py)
    data = [
        [0.3275, 0.0000696600, 95.01],  # Standard PCA
        [2.8450, 0.0000732804, 95.00],  # Incremental PCA
        [107.4375, 0.0013919449, ],  # Kernel PCA (RBF)
        [353.3227, 0.0004158161, ]   # Sparse PCA
    ]
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame(data, index=variants, columns=metrics).T
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.axis('tight')

    # Create the table
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=["#404040"] * 4,
        rowColours=["#f2f2f2"] * 3
    )

    # Styling the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)

    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Column headers
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4F81BD') # Professional Blue
        elif col == -1: # Row headers
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#D9E1F2')
        
        # Zebra striping for rows
        elif row % 2 == 0:
            cell.set_facecolor('#F2F2F2')

    plt.title("Dimensionality Reduction Variant Comparison", fontsize=16, pad=20, weight='bold')
    
    # Save the image
    output_path = "pca_variant_comparison.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Comparison table saved to: {output_path}")

if __name__ == "__main__":
    generate_comparison_table()
