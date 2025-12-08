import os
from preprocessing import load_full_dataset, split_dataset, make_dataloaders, plot_class_distribution, find_corrupted_images, show_grid

data_dir = os.path.join("..", "dataset", "1", "HMU-GC-HE-30K", "all_image")

# Load full dataset
dataset = load_full_dataset(data_dir)

# Plot class distribution
plot_class_distribution(dataset)

# Check for corrupted images
corrupted = find_corrupted_images(dataset)
print(f"Found {len(corrupted)} corrupted images")

# Visualize samples per class
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
show_grid(dataset, idx_to_class, samples_per_class=4)

