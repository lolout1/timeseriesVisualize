import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import torch

# --------------------------------------------------------------------------------
# 1. BUTTERWORTH FILTER (CPU Implementation)
# --------------------------------------------------------------------------------
def butterworth_filter(data: np.ndarray, cutoff: float = 7.0, fs: float = 32.0, order: int = 6) -> np.ndarray:
    """
    Applies a Butterworth low-pass filter to the data.
    
    Parameters:
    -----------
    data    : (T, D) ndarray of raw data (T timesteps, D dimensions).
    cutoff  : Cutoff frequency for the low-pass filter.
    fs      : Sampling frequency (Hz).
    order   : Filter order.
    
    Returns:
    --------
    filtered : (T, D) ndarray of filtered data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply filtfilt for zero-phase filtering
    filtered = filtfilt(b, a, data, axis=0, padlen=3*(max(len(b), len(a)) - 1))
    return filtered

# --------------------------------------------------------------------------------
# 2. HYBRID KALMAN FILTER (GPU Implementation with PyTorch)
# --------------------------------------------------------------------------------
def _apply_hybrid_kalman(window: np.ndarray, fs: float = 32.0, device: torch.device = torch.device('cuda:0')) -> np.ndarray:
    """
    Applies a hybrid Kalman filter:
    - Some "critical" joints (e.g., head, pelvis, knees) get a JOINT-WISE Kalman filter.
    - The rest of the joints get a SINGLE, GLOBAL Kalman filter to maintain inter-joint consistency.
    
    Parameters:
    -----------
    window : (T, 96) 
        A single sliding-window of skeleton data with T timesteps and 32 joints × 3 coords each.
    fs : float
        Sampling frequency for building the state transition, etc.
    device : torch.device
        The CUDA device to use for computations.
    
    Returns:
    --------
    kalman_filtered : (T, 96)
        The final Kalman-filtered data for this window, same shape as input.
    """
    
    # Define critical joint indices (0-based)
    CRITICAL_JOINTS = [3, 2, 13, 14, 27, 28]  # Example indices for HEAD, NECK, KNEES, etc.
    
    # Initialize Kalman filter parameters
    Q = 0.001  # Process noise covariance
    R = 0.01   # Measurement noise covariance
    
    T, D = window.shape  # T timesteps, D=96 dimensions
    kalman_filtered = torch.zeros((T, D), device=device)
    
    window_tensor = torch.tensor(window, dtype=torch.float32, device=device)
    
    # --------------------
    # Joint-Wise Kalman Filtering for Critical Joints
    # --------------------
    for j in CRITICAL_JOINTS:
        for axis in range(3):  # x, y, z
            idx = j * 3 + axis
            measurements = window_tensor[:, idx]
            
            # Initialize state
            x = torch.zeros(T, device=device)
            P = torch.zeros(T, device=device)
            x[0] = measurements[0]
            P[0] = 1.0
            
            for k in range(1, T):
                # Predict
                x_pred = x[k-1]
                P_pred = P[k-1] + Q
                
                # Update
                K = P_pred / (P_pred + R)
                x[k] = x_pred + K * (measurements[k] - x_pred)
                P[k] = (1 - K) * P_pred
            
            kalman_filtered[:, idx] = x
    
    # --------------------
    # Global Kalman Filtering for Non-Critical Joints
    # --------------------
    all_joints = set(range(32))
    non_critical_joints = list(all_joints - set(CRITICAL_JOINTS))
    
    non_critical_indices = []
    for j in non_critical_joints:
        non_critical_indices.extend([j*3, j*3+1, j*3+2])
    
    if non_critical_indices:
        # Extract non-critical joint data
        measurements = window_tensor[:, non_critical_indices]  # Shape: (T, N*3)
        
        # Initialize state and covariance
        x = torch.zeros_like(measurements)
        P = torch.ones_like(measurements)
        
        x[0] = measurements[0]
        P[0] = 1.0
        
        for k in range(1, T):
            # Predict
            x_pred = x[k-1]
            P_pred = P[k-1] + Q
            
            # Update
            K = P_pred / (P_pred + R)
            x[k] = x_pred + K * (measurements[k] - x_pred)
            P[k] = (1 - K) * P_pred
        
        kalman_filtered[:, non_critical_indices] = x
    
    # Move data back to CPU and convert to numpy
    kalman_filtered_cpu = kalman_filtered.cpu().numpy()
    
    return kalman_filtered_cpu

# --------------------------------------------------------------------------------
# 3. DATASET BUILDER CLASS
# --------------------------------------------------------------------------------
class DatasetBuilder:
    def __init__(self, dataset: object, window_size: int = 128, stride: int = 64,
                 task: str = 'fd', fs: float = 32.0, output_root: str = "visualizations", **kwargs) -> None:
        """
        Initializes the DatasetBuilder with sliding window parameters.
        """
        self.dataset = dataset
        self.window_size = window_size  # e.g., 4 seconds @ 32 Hz
        self.stride = stride  # 50% overlap
        self.task = task
        self.fs = fs
        self.kwargs = kwargs
        self.output_root = output_root

        os.makedirs(self.output_root, exist_ok=True)

        # Set device for GPU acceleration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {self.device}")

    def process_skeleton_data(self, subjects: list):
        """
        Processes skeleton data with sliding windows, then creates all visualizations:
        1. raw vs. filtered vs. normalized (comparison)
        2. zoomed-in normalized
        3. new: Kalman vs. (filtered+normalized)
        """
        print("\n[INFO] Processing skeleton data with sliding windows...")
        for trial in self.dataset.matched_trials:
            if trial.subject_id not in subjects:
                continue

            trial_id = f"S{trial.subject_id:02d}A{trial.action_id}T{trial.sequence_number:02d}"
            activity_dir = os.path.join(self.output_root, f"A{trial.action_id}", f"S{trial.subject_id:02d}")
            os.makedirs(activity_dir, exist_ok=True)

            try:
                # 1. Load raw data
                raw_data = np.loadtxt(trial.files['skeleton'], delimiter=',')
                if raw_data.ndim == 1:
                    raw_data = raw_data.reshape(-1, 1)  # Ensure 2D shape
                print(f"[DEBUG] Trial {trial_id}: raw_data.shape = {raw_data.shape}")

                # 2. Sliding Window
                num_samples = raw_data.shape[0]
                windows = []
                if num_samples >= self.window_size:
                    for start in range(0, num_samples - self.window_size + 1, self.stride):
                        end = start + self.window_size
                        window = raw_data[start:end, :]
                        if window.ndim == 1:
                            window = window.reshape(-1, 1)
                        windows.append((start, end, window))
                else:
                    print(f"[WARNING] Trial {trial_id} has insufficient samples ({num_samples}) for windowing.")
                    continue

                # 3. Visualizations per trial
                if len(windows) > 0:
                    self._save_comparison_visualization(trial_id, activity_dir, windows)
                    self._save_detailed_normalized_visualization(trial_id, activity_dir, windows)
                    self._save_kalman_visualization(trial_id, activity_dir, windows)
                    print(f"[INFO] Processed and visualized Trial {trial_id} with {len(windows)} windows")
                else:
                    print(f"[WARNING] No windows generated for Trial {trial_id}. Skipping visualization.")

            except Exception as e:
                print(f"[ERROR] Processing skeleton data for Trial {trial_id}: {e}")
                continue

    # --------------------------------------------------------------------------------
    # 3.1 COMPARISON VISUALIZATION (Raw vs. Butterworth vs. Normalized)
    # --------------------------------------------------------------------------------
    def _save_comparison_visualization(self, trial_id: str, activity_dir: str, windows: list):
        """
        Saves raw, Butterworth-filtered, and Butterworth + normalized skeleton data visualizations.
        Each column = one window.
        Rows = [raw, filtered, normalized].
        """
        try:
            num_windows = len(windows)
            fig, axes = plt.subplots(3, num_windows, figsize=(15, 10), sharey=True)

            # Ensure `axes` is a 2D array even if there's only one window
            if num_windows == 1:
                axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

            for idx, (_, _, window) in enumerate(windows):
                # Apply Butterworth filter on CPU
                filtered = butterworth_filter(window, cutoff=7.0, fs=self.fs)
                normalized = (filtered - np.mean(filtered, axis=0)) / (np.std(filtered, axis=0) + 1e-5)

                # Plot raw data
                axes[0, idx].plot(window)
                axes[0, idx].set_title(f"Window {idx}")
                if idx == 0:
                    axes[0, idx].set_ylabel("Raw")

                # Plot filtered data
                axes[1, idx].plot(filtered)
                if idx == 0:
                    axes[1, idx].set_ylabel("Filtered")

                # Plot normalized data
                axes[2, idx].plot(normalized)
                if idx == 0:
                    axes[2, idx].set_ylabel("Norm+Filt")

            plt.tight_layout()
            comparison_path = os.path.join(activity_dir, f"{trial_id}_comparison.png")
            plt.savefig(comparison_path, dpi=150)
            plt.close()
            print(f"[INFO] Saved comparison visualization for Trial {trial_id} at {comparison_path}")

        except Exception as e:
            print(f"[ERROR] Visualizing windows for Trial {trial_id}: {e}")

    # --------------------------------------------------------------------------------
    # 3.2 DETAILED NORMALIZED VISUALIZATION (Zoomed-In Normalized)
    # --------------------------------------------------------------------------------
    def _save_detailed_normalized_visualization(self, trial_id: str, activity_dir: str, windows: list):
        """
        Saves a single figure with subplots for each window, showing only
        the Butterworth-filtered + normalized data for a more detailed view.
        """
        try:
            num_windows = len(windows)
            fig, axes = plt.subplots(1, num_windows, figsize=(15, 5), sharey=True)

            # Ensure `axes` is a 1D array even if there's only one window
            if num_windows == 1:
                axes = [axes]

            for idx, (_, _, window) in enumerate(windows):
                # Apply Butterworth filter on CPU
                filtered = butterworth_filter(window, cutoff=7.0, fs=self.fs)
                normalized = (filtered - np.mean(filtered, axis=0)) / (np.std(filtered, axis=0) + 1e-5)

                axes[idx].plot(normalized)
                axes[idx].set_title(f"Window {idx}")

            plt.tight_layout()
            detailed_path = os.path.join(activity_dir, f"{trial_id}_all_windows.png")
            plt.savefig(detailed_path, dpi=150)
            plt.close()
            print(f"[INFO] Saved trial visualization for Trial {trial_id} at {detailed_path}")

        except Exception as e:
            print(f"[ERROR] Visualizing detailed normalized windows for Trial {trial_id}: {e}")

    # --------------------------------------------------------------------------------
    # 3.3 NEW: KALMAN VISUALIZATION (Butterworth+Normalized vs. Hybrid-Kalman)
    # --------------------------------------------------------------------------------
    def _save_kalman_visualization(self, trial_id: str, activity_dir: str, windows: list):
        """
        For each window, create separate .png files that compare:
          - Butterworth+Normalized data
          - Hybrid Kalman filtered data
        Each axis (x, y, z) is plotted separately in individual subplots.
        """
        try:
            for idx, (_, _, window) in enumerate(windows):
                # 1. Butterworth + Normalized
                filtered = butterworth_filter(window, cutoff=7.0, fs=self.fs)
                normalized = (filtered - np.mean(filtered, axis=0)) / (np.std(filtered, axis=0) + 1e-5)

                # 2. Hybrid Kalman on GPU
                kalman_output = _apply_hybrid_kalman(normalized, fs=self.fs, device=self.device)

                # 3. Plot each axis separately
                # We'll plot all 96 dimensions, but for clarity, you might want to plot key joints
                # Here, we'll plot a subset for demonstration (e.g., first 9 dimensions: 3 joints)
                # Adjust as needed for full detail

                # For detailed analysis, plotting all dimensions might be too cluttered
                # Instead, iterate through each joint's x, y, z
                num_joints = 32
                num_axes = 3
                total_plots = num_joints * num_axes

                # To prevent excessive memory usage, consider plotting a selected subset
                # For full 96 plots, you might need to adjust figure size accordingly
                # Here, we'll plot all axes in a grid layout

                fig, axes = plt.subplots(num_joints, num_axes, figsize=(15, num_joints * 3), sharex=True)
                plt.subplots_adjust(hspace=0.5)

                for joint in range(num_joints):
                    for axis in range(num_axes):
                        dim = joint * 3 + axis
                        ax = axes[joint, axis] if num_joints > 1 else axes[axis]
                        ax.plot(normalized[:, dim], label="Butterworth+Normalized", alpha=0.7)
                        ax.plot(kalman_output[:, dim], label="Hybrid Kalman", alpha=0.7, linestyle='--')
                        ax.set_title(f"Joint {joint+1} Axis {'XYZ'[axis]}")
                        if joint == 0:
                            ax.set_ylabel(f"{'XYZ'[axis]}")
                        if joint == num_joints - 1:
                            ax.set_xlabel("Time (samples)")
                        ax.legend(loc='upper right', fontsize='small')

                plt.tight_layout()
                kalman_path = os.path.join(activity_dir, f"{trial_id}_window{idx}_kalman.png")
                plt.savefig(kalman_path, dpi=200)
                plt.close()

            print(f"[INFO] Saved Kalman comparison visualizations for Trial {trial_id}")

        except Exception as e:
            print(f"[ERROR] Visualizing Kalman windows for Trial {trial_id}: {e}")

# --------------------------------------------------------------------------------
# 4. MAIN SCRIPT
# --------------------------------------------------------------------------------
def main():
    """
    Example main function demonstrating how to use the DatasetBuilder on the SmartFallMM dataset.
    """
    from dataset import SmartFallMM  # Adjust to your actual dataset import

    data_root = os.path.join(os.getcwd(), "data", "smartfallmm")
    output_root = os.path.join(os.getcwd(), "visualizations")

    # 4.1 Initialize dataset object
    dataset = SmartFallMM(root_dir=data_root)
    dataset.add_modality("young", "skeleton")

    print("\n[INFO] Loading and matching files...")
    dataset.load_files()
    dataset.match_trials()

    # 4.2 Initialize DatasetBuilder with typical window settings
    builder = DatasetBuilder(
        dataset=dataset,
        window_size=128,
        stride=64,
        task="fd",
        fs=32.0,
        output_root=output_root
    )

    # 4.3 Define subjects for processing
    subjects = list(range(29, 47))
    print(f"\n[INFO] Processing dataset for subjects: {subjects}")

    # 4.4 Process skeleton data with all visualizations, including new Kalman
    builder.process_skeleton_data(subjects)


if __name__ == "__main__":
    main()
