import os
import torch
import matplotlib.pyplot as plt

def main():
    # Folder where your 16 .pth files are located
    checkpoint_dir = "checkpoints" 
    
    epochs = []
    psnr_c = []
    psnr_s = []

    # Find and load all .pth files
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Folder '{checkpoint_dir}' not found.")
        return
        
    pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    
    if not pth_files:
        print(f"No .pth files found in '{checkpoint_dir}'.")
        return
        
    print(f"Found {len(pth_files)} checkpoint files. Extracting data...")

    for file_name in pth_files:
        file_path = os.path.join(checkpoint_dir, file_name)
        try:
            # Load checkpoint safely to CPU
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
            
            # Extract the exact keys saved in your train.py
            if "epoch" in checkpoint and "avg_psnr_c" in checkpoint and "avg_psnr_s" in checkpoint:
                epochs.append(checkpoint["epoch"])
                psnr_c.append(checkpoint["avg_psnr_c"])
                psnr_s.append(checkpoint["avg_psnr_s"])
        except Exception as e:
            print(f"Could not read {file_name}: {e}")

    if not epochs:
        print("No valid metric data found in the .pth files.")
        return

    # Sort data sequentially by epoch (10, 20, 30... 160)
    sorted_data = sorted(zip(epochs, psnr_c, psnr_s))
    epochs, psnr_c, psnr_s = zip(*sorted_data)

    # --- Plotting the Data (Professional Academic Style) ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    ax.grid(True, linestyle='-', alpha=0.3, color='gray')

    # Plot lines with distinct markers
    ax.plot(epochs, psnr_c, color='#1f77b4', linewidth=2.5, marker='o', markersize=6, label='PSNR-C (Cover Image Quality)')
    ax.plot(epochs, psnr_s, color='#ff7f0e', linewidth=2.5, marker='s', markersize=6, label='PSNR-S (Secret Recovery Quality)')

    # Labels & Titles
    ax.set_xlabel("Epochs", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    plt.title("Watermarking PSNR Progression (160 Epochs)", fontsize=16, fontweight='bold', pad=15)
    
    # Legend formatting
    ax.legend(loc='lower right', frameon=True, fontsize=11, framealpha=0.9, edgecolor='#dddddd')
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#333333')

    # Render and Save
    plt.tight_layout()
    plt.savefig('psnr_progression_160_epochs.png', dpi=300, bbox_inches='tight')
    print("Successfully generated and saved 'psnr_progression_160_epochs.png'")

if __name__ == "__main__":
    main()