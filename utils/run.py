import os
from dataset import SmartFallMM  # Replace with actual import if different
from loader import DatasetBuilder

def main():
    data_root = os.path.join(os.getcwd(), "data", "smartfallmm")  # Adjust as needed
    output_root = os.path.join(os.getcwd(), "visualizations")  # Adjust as needed

    # Initialize your dataset object (replace with actual implementation)
    dataset = SmartFallMM(root_dir=data_root)
    dataset.add_modality("young", "accelerometer", ["phone", "watch"])
    dataset.add_modality("old", "accelerometer", ["phone", "watch"])

    print("\n[INFO] Loading and matching files...")
    dataset.load_files()
    dataset.match_trials()

    # Initialize DatasetBuilder
    builder = DatasetBuilder(dataset=dataset, mode="avg_pool", max_length=94, task="fd", fs=31.25)

    # Define subjects to process
    subjects = list(range(29, 47))
    print(f"\n[INFO] Building dataset for subjects: {subjects}")

    # Process with Median + Butterworth
    builder.make_dataset_with_med(subjects=subjects)

    # Normalize with Median + Butterworth
    builder.normalize_with_med()

    # Process with Butterworth Only
    builder.make_dataset_without_med(subjects=subjects)

    # Normalize with Butterworth Only
    builder.normalize_without_med()

    # Generate and save all visualizations
    builder.visualize_and_save_all(output_root=output_root)

if __name__ == "__main__":
    main()