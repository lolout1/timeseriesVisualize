"""
File: loader.py

Description:
    - Contains definitions for:
       1) fall_detection_filter (median + Butterworth)
       2) butterworth_filter (Butterworth Only)
       3) DatasetBuilder class that:
          - Loads raw/filtered/normalized accelerometer data
          - Provides multiple visualization methods:
             a) visualize_subject_activity (raw vs. filter+norm)
             b) visualize_subject_activity_normonly (norm-only phone vs. watch)
             c) visualize_activity_average (per-activity average)
             d) create_all_activities_summary (combine all activities)
             e) visualize_activity_subject_averages_in_same_subplot (filtered+norm averages)
             f) visualize_activity_all_trials_same_subplot (filtered+norm, phone+watch overlaid)
             g) visualize_trial_normxyz_smv (per-trial normalized X,Y,Z and SMV plots)
             h) visualize_trial_comparison (compare both filtering pipelines)
"""
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt
from sklearn.preprocessing import StandardScaler

from processor.base import Processor  # Ensure this path is correct


# ---------------- FILTERING FUNCTIONS ---------------- #
def fall_detection_filter(data: np.ndarray, fs: float = 31.25) -> np.ndarray:
    """
    A two-stage filtering approach optimized for ~31.25Hz accelerometer data:
      1) Median filter (window=5)
      2) Low-pass Butterworth (cutoff ~7Hz, order=6)

    Parameters:
    - data: Input signal array (N, 3), columns => x, y, z
    - fs: Sampling frequency in Hz

    Returns:
    - Filtered signal array (N, 3)
    """
    # 1) Median filter
    filtered_data = np.zeros_like(data)
    for axis in range(3):
        filtered_data[:, axis] = medfilt(data[:, axis], kernel_size=5)  # Kernel size = 5

    # 2) Low-pass Butterworth
    order = 6
    cutoff = 7.0  # Hz
    nyquist = fs * 0.5
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    pad_size = order * 4
    padded = np.pad(filtered_data, ((pad_size, pad_size), (0, 0)), mode='edge')

    final_filtered = np.zeros_like(padded)
    for axis in range(3):
        final_filtered[:, axis] = filtfilt(b, a, padded[:, axis])

    return final_filtered[pad_size:-pad_size, :]


def butterworth_filter(data: np.ndarray, cutoff: float = 7.0, fs: float = 31.25, order: int = 6) -> np.ndarray:
    """
    Applies only a Butterworth low-pass filter to the data.

    Parameters:
    - data: Input signal array (N, 3)
    - cutoff: Cutoff frequency in Hz
    - fs: Sampling frequency in Hz
    - order: Order of the Butterworth filter

    Returns:
    - Filtered signal array (N, 3)
    """
    nyquist = fs * 0.5
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    pad_size = order * 4
    padded = np.pad(data, ((pad_size, pad_size), (0, 0)), mode='edge')

    final_filtered = np.zeros_like(padded)
    for axis in range(3):
        final_filtered[:, axis] = filtfilt(b, a, padded[:, axis])

    return final_filtered[pad_size:-pad_size, :]


# ---------------- DATASET BUILDER ---------------- #
class DatasetBuilder:
    """
    Builds a dataset from matched trials, storing raw, filtered (two pipelines), normalized data.
    Provides multiple visualization methods.
    """

    def __init__(self, dataset: object, mode: str, max_length: int, task: str = 'fd', fs: float = 31.25, **kwargs) -> None:
        self.dataset = dataset
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fs = fs  # Sampling frequency
        self.kwargs = kwargs

        self.data_with_med: Dict[str, np.ndarray] = {}
        self.data_without_med: Dict[str, np.ndarray] = {}
        self.labels_with_med: np.ndarray = np.array([])
        self.labels_without_med: np.ndarray = np.array([])
        self.successful_indices_with_med: List[int] = []  # Track successful trials for with_med
        self.successful_indices_without_med: List[int] = []  # Track successful trials for without_med

        self._raw_data_with_med: Dict[str, List[np.ndarray]] = {}
        self._raw_data_without_med: Dict[str, List[np.ndarray]] = {}
        self._filtered_with_med: Dict[str, List[np.ndarray]] = {}
        self._filtered_without_med: Dict[str, List[np.ndarray]] = {}
        self._normalized_with_med: Dict[str, np.ndarray] = {}
        self._normalized_without_med: Dict[str, np.ndarray] = {}

    # ---------------- MAKE DATASET WITH MEDIAN + BUTTERWORTH ---------------- #
    def make_dataset_with_med(self, subjects: List[int]) -> None:
        """
        Load data for specified subjects, storing raw & filtered signals using Median + Butterworth filter.
        """
        print(f"\n[INFO] Making dataset with Median + Butterworth for subjects: {subjects}")
        trial_subjects = {t.subject_id for t in self.dataset.matched_trials}
        matching = set(subjects) & trial_subjects

        print("\n[INFO] Subject statistics:")
        print(f"Subjects in matched trials: {sorted(list(trial_subjects))}")
        print(f"Requested subjects: {sorted(list(subjects))}")
        print(f"Matching subjects: {sorted(list(matching))}")

        if not matching:
            print("\n[WARNING] No matching subjects found for Median + Butterworth pipeline!")
            return

        needs_cleaning_dir = os.path.join(self.dataset.root_dir, 'needs_cleaning')
        os.makedirs(needs_cleaning_dir, exist_ok=True)
        print(f"[INFO] Using needs_cleaning directory: {needs_cleaning_dir}")

        # Initialize raw and filtered data storage
        self._raw_data_with_med = {'labels': []}
        self._filtered_with_med = {}

        processed = 0
        for idx, trial in enumerate(self.dataset.matched_trials):
            if trial.subject_id not in matching:
                continue

            label = self._compute_label(trial)
            processed += 1
            print(f"[INFO] Processing (with_med) trial: S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}")

            success = True

            for mod_key, file_path in trial.files.items():
                try:
                    # 1) Read raw
                    processor = Processor(file_path, self.mode, self.max_length)
                    raw_arr = processor.process()

                    # store raw
                    if mod_key not in self._raw_data_with_med:
                        self._raw_data_with_med[mod_key] = []
                    self._raw_data_with_med[mod_key].append(raw_arr)

                    # 2a) Filter with Median + Butterworth
                    filt_with_med = fall_detection_filter(raw_arr, fs=self.fs)
                    if mod_key not in self._filtered_with_med:
                        self._filtered_with_med[mod_key] = []
                    self._filtered_with_med[mod_key].append(filt_with_med)
                except Exception as e:
                    print(f"[ERROR] (with_med) Error processing file {file_path}: {e}")
                    self._copy_to_needs_cleaning(needs_cleaning_dir, file_path)
                    success = False
                    break

            if success:
                self.successful_indices_with_med.append(idx)
                self._raw_data_with_med['labels'].append(label)
                for mk, arr_ in self._filtered_with_med.items():
                    self.data_with_med.setdefault(mk, []).append(arr_[-1])  # Append the latest filtered data
                self.labels_with_med = np.array(self._raw_data_with_med['labels'])

        print(f"\n[INFO] (with_med) Processed {processed} trials")
        print(f"[INFO] (with_med) Total trials processed successfully: {len(self.successful_indices_with_med)}")

        if not self.labels_with_med.size:
            print("\n[WARNING] No data was loaded for Median + Butterworth pipeline. Exiting dataset preparation.")
            return

        print("\n[INFO] Converting Median + Butterworth data to numpy arrays...")
        for k, arr_list in self.data_with_med.items():
            if k == 'labels':
                continue
            try:
                stacked = np.stack(arr_list, axis=0)
                self.data_with_med[k] = stacked
                print(f"[INFO] (with_med) {k} shape: {stacked.shape}")
            except Exception as e:
                print(f"[ERROR] (with_med) Error stacking {k}: {e}")

    # ---------------- MAKE DATASET WITH BUTTERWORTH ONLY ---------------- #
    def make_dataset_without_med(self, subjects: List[int]) -> None:
        """
        Load data for specified subjects, storing raw & filtered signals using Butterworth Only filter.
        """
        print(f"\n[INFO] Making dataset with Butterworth Only for subjects: {subjects}")
        trial_subjects = {t.subject_id for t in self.dataset.matched_trials}
        matching = set(subjects) & trial_subjects

        print("\n[INFO] Subject statistics:")
        print(f"Subjects in matched trials: {sorted(list(trial_subjects))}")
        print(f"Requested subjects: {sorted(list(subjects))}")
        print(f"Matching subjects: {sorted(list(matching))}")

        if not matching:
            print("\n[WARNING] No matching subjects found for Butterworth Only pipeline!")
            return

        needs_cleaning_dir = os.path.join(self.dataset.root_dir, 'needs_cleaning')
        os.makedirs(needs_cleaning_dir, exist_ok=True)
        print(f"[INFO] Using needs_cleaning directory: {needs_cleaning_dir}")

        # Initialize raw and filtered data storage
        self._raw_data_without_med = {'labels': []}
        self._filtered_without_med = {}

        processed = 0
        for idx, trial in enumerate(self.dataset.matched_trials):
            if trial.subject_id not in matching:
                continue

            label = self._compute_label(trial)
            processed += 1
            print(f"[INFO] Processing (without_med) trial: S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}")

            success = True

            for mod_key, file_path in trial.files.items():
                try:
                    # 1) Read raw
                    processor = Processor(file_path, self.mode, self.max_length)
                    raw_arr = processor.process()

                    # store raw
                    if mod_key not in self._raw_data_without_med:
                        self._raw_data_without_med[mod_key] = []
                    self._raw_data_without_med[mod_key].append(raw_arr)

                    # 2a) Filter with Butterworth Only
                    filt_without_med = butterworth_filter(raw_arr, cutoff=7.0, fs=self.fs, order=6)
                    if mod_key not in self._filtered_without_med:
                        self._filtered_without_med[mod_key] = []
                    self._filtered_without_med[mod_key].append(filt_without_med)
                except Exception as e:
                    print(f"[ERROR] (without_med) Error processing file {file_path}: {e}")
                    self._copy_to_needs_cleaning(needs_cleaning_dir, file_path)
                    success = False
                    break

            if success:
                self.successful_indices_without_med.append(idx)
                self._raw_data_without_med['labels'].append(label)
                for mk, arr_ in self._filtered_without_med.items():
                    self.data_without_med.setdefault(mk, []).append(arr_[-1])  # Append the latest filtered data
                self.labels_without_med = np.array(self._raw_data_without_med['labels'])

        print(f"\n[INFO] (without_med) Processed {processed} trials")
        print(f"[INFO] (without_med) Total trials processed successfully: {len(self.successful_indices_without_med)}")

        if not self.labels_without_med.size:
            print("\n[WARNING] No data was loaded for Butterworth Only pipeline. Exiting dataset preparation.")
            return

        print("\n[INFO] Converting Butterworth Only data to numpy arrays...")
        for k, arr_list in self.data_without_med.items():
            if k == 'labels':
                continue
            try:
                stacked = np.stack(arr_list, axis=0)
                self.data_without_med[k] = stacked
                print(f"[INFO] (without_med) {k} shape: {stacked.shape}")
            except Exception as e:
                print(f"[ERROR] (without_med) Error stacking {k}: {e}")

    # ---------------- NORMALIZE WITH MEDIAN + BUTTERWORTH ---------------- #
    def normalize_with_med(self) -> None:
        """
        Normalizes the Median + Butterworth filtered data and stores in self._normalized_with_med.
        """
        print("\n[INFO] Normalizing data with Median + Butterworth filtering...")
        self._normalized_with_med.clear()
        for k, arr in self.data_with_med.items():
            if k == 'labels':
                continue
            print(f"[INFO] Normalizing (with_med) {k}...")
            shape_ = arr.shape
            flattened = arr.reshape(-1, shape_[-1])

            scaler = StandardScaler()
            norm_flat = scaler.fit_transform(flattened + 1e-10)  # Adding epsilon to avoid division by zero
            norm_arr = norm_flat.reshape(shape_)
            self._normalized_with_med[k] = norm_arr
            print(f"[INFO] (with_med) {k} normalized shape: {norm_arr.shape}")

    # ---------------- NORMALIZE WITH BUTTERWORTH ONLY ---------------- #
    def normalize_without_med(self) -> None:
        """
        Normalizes the Butterworth Only filtered data and stores in self._normalized_without_med.
        """
        print("\n[INFO] Normalizing data with Butterworth Only filtering...")
        self._normalized_without_med.clear()
        for k, arr in self.data_without_med.items():
            if k == 'labels':
                continue
            print(f"[INFO] Normalizing (without_med) {k}...")
            shape_ = arr.shape
            flattened = arr.reshape(-1, shape_[-1])

            scaler = StandardScaler()
            norm_flat = scaler.fit_transform(flattened + 1e-10)
            norm_arr = norm_flat.reshape(shape_)
            self._normalized_without_med[k] = norm_arr
            print(f"[INFO] (without_med) {k} normalized shape: {norm_arr.shape}")

    # ---------------- VISUALIZATION METHODS ---------------- #
    # The visualization methods have been updated to handle axes correctly.

    def visualize_and_save_all(self, output_root: str) -> None:
        """
        Generates and saves all visualizations for both filtering pipelines.
        """
        print(f"\n[INFO] Generating and saving visualizations to '{output_root}'...")
        os.makedirs(output_root, exist_ok=True)

        from collections import defaultdict
        trial_map_with_med = defaultdict(list)
        trial_map_without_med = defaultdict(list)

        # Group by (activity, subject) for with_med
        for relative_idx, global_idx in enumerate(self.successful_indices_with_med):
            trial = self.dataset.matched_trials[global_idx]
            trial_map_with_med[(trial.action_id, trial.subject_id)].append(relative_idx)

        # Group by (activity, subject) for without_med
        for relative_idx, global_idx in enumerate(self.successful_indices_without_med):
            trial = self.dataset.matched_trials[global_idx]
            trial_map_without_med[(trial.action_id, trial.subject_id)].append(relative_idx)

        # Distinct activity IDs
        activity_set_with_med = set(act_id for (act_id, _) in trial_map_with_med.keys())
        activity_set_without_med = set(act_id for (act_id, _) in trial_map_without_med.keys())
        all_activities = sorted(activity_set_with_med | activity_set_without_med)

        print("\n[INFO] Generating visualizations for Median + Butterworth pipeline...")
        for act_id in all_activities:
            activity_dir = os.path.join(output_root, f"A{act_id}")
            os.makedirs(activity_dir, exist_ok=True)

            # Subjects for this activity in with_med
            subject_ids_with_med = sorted([sub_id for (a, sub_id) in trial_map_with_med.keys() if a == act_id])

            for subj_id in subject_ids_with_med:
                subject_dir = os.path.join(activity_dir, f"S{subj_id:02d}")
                os.makedirs(subject_dir, exist_ok=True)

                idx_list = trial_map_with_med.get((act_id, subj_id), [])
                idx_list_sorted = sorted(idx_list, key=lambda x: self.dataset.matched_trials[self.successful_indices_with_med[x]].sequence_number)

                # a) Raw vs. Filter+Norm
                fig1 = self.visualize_subject_activity(idx_list_sorted, subj_id, act_id, pipeline='with_med')
                if fig1:
                    raw_filter_file = os.path.join(subject_dir, f"S{subj_id:02d}_raw_filter_with_med.png")
                    fig1.savefig(raw_filter_file, dpi=150)
                    plt.close(fig1)
                    print(f"[INFO] Saved {raw_filter_file}")

                # b) Norm-only phone vs watch
                fig2 = self.visualize_subject_activity_normonly(idx_list_sorted, subj_id, act_id, pipeline='with_med')
                if fig2:
                    normonly_file = os.path.join(subject_dir, f"S{subj_id:02d}_normonly_with_med.png")
                    fig2.savefig(normonly_file, dpi=150)
                    plt.close(fig2)
                    print(f"[INFO] Saved {normonly_file}")

                # c) Per-trial normalized X,Y,Z and SMV
                for relative_idx in idx_list_sorted:
                    global_idx = self.successful_indices_with_med[relative_idx]
                    trial_obj = self.dataset.matched_trials[global_idx]
                    seq_num = trial_obj.sequence_number

                    # Per-Trial Normalized Plots
                    fig_xyz_med, fig_smv_med = self.visualize_trial_normxyz_smv(global_idx, pipeline='with_med')
                    if fig_xyz_med and fig_smv_med:
                        out_xyz_med = os.path.join(subject_dir, f"T{seq_num:02d}_normxyz_with_med.png")
                        out_smv_med = os.path.join(subject_dir, f"T{seq_num:02d}_normsmv_with_med.png")
                        fig_xyz_med.savefig(out_xyz_med, dpi=150)
                        fig_smv_med.savefig(out_smv_med, dpi=150)
                        plt.close(fig_xyz_med)
                        plt.close(fig_smv_med)
                        print(f"[INFO] Saved {out_xyz_med} and {out_smv_med}")

                # d) Subject Averages in Same Subplot
                fig_avg_med = self.visualize_activity_subject_averages_in_same_subplot(act_id, [subj_id], pipeline='with_med')
                if fig_avg_med:
                    avg_file_med = os.path.join(subject_dir, f"S{subj_id:02d}_average_with_med.png")
                    fig_avg_med.savefig(avg_file_med, dpi=150)
                    plt.close(fig_avg_med)
                    print(f"[INFO] Saved {avg_file_med}")

        print("\n[INFO] Generating visualizations for Butterworth Only pipeline...")
        for act_id in all_activities:
            activity_dir = os.path.join(output_root, f"A{act_id}")
            os.makedirs(activity_dir, exist_ok=True)

            # Subjects for this activity in without_med
            subject_ids_without_med = sorted([sub_id for (a, sub_id) in trial_map_without_med.keys() if a == act_id])

            for subj_id in subject_ids_without_med:
                subject_dir = os.path.join(activity_dir, f"S{subj_id:02d}")
                os.makedirs(subject_dir, exist_ok=True)

                idx_list = trial_map_without_med.get((act_id, subj_id), [])
                idx_list_sorted = sorted(idx_list, key=lambda x: self.dataset.matched_trials[self.successful_indices_without_med[x]].sequence_number)

                # a) Raw vs. Filter+Norm
                fig1 = self.visualize_subject_activity(idx_list_sorted, subj_id, act_id, pipeline='without_med')
                if fig1:
                    raw_filter_file = os.path.join(subject_dir, f"S{subj_id:02d}_raw_filter_without_med.png")
                    fig1.savefig(raw_filter_file, dpi=150)
                    plt.close(fig1)
                    print(f"[INFO] Saved {raw_filter_file}")

                # b) Norm-only phone vs watch
                fig2 = self.visualize_subject_activity_normonly(idx_list_sorted, subj_id, act_id, pipeline='without_med')
                if fig2:
                    normonly_file = os.path.join(subject_dir, f"S{subj_id:02d}_normonly_without_med.png")
                    fig2.savefig(normonly_file, dpi=150)
                    plt.close(fig2)
                    print(f"[INFO] Saved {normonly_file}")

                # c) Per-trial normalized X,Y,Z and SMV
                for relative_idx in idx_list_sorted:
                    global_idx = self.successful_indices_without_med[relative_idx]
                    trial_obj = self.dataset.matched_trials[global_idx]
                    seq_num = trial_obj.sequence_number

                    # Per-Trial Normalized Plots
                    fig_xyz_bw, fig_smv_bw = self.visualize_trial_normxyz_smv(global_idx, pipeline='without_med')
                    if fig_xyz_bw and fig_smv_bw:
                        out_xyz_bw = os.path.join(subject_dir, f"T{seq_num:02d}_normxyz_without_med.png")
                        out_smv_bw = os.path.join(subject_dir, f"T{seq_num:02d}_normsmv_without_med.png")
                        fig_xyz_bw.savefig(out_xyz_bw, dpi=150)
                        fig_smv_bw.savefig(out_smv_bw, dpi=150)
                        plt.close(fig_xyz_bw)
                        plt.close(fig_smv_bw)
                        print(f"[INFO] Saved {out_xyz_bw} and {out_smv_bw}")

                # d) Subject Averages in Same Subplot
                fig_avg_bw = self.visualize_activity_subject_averages_in_same_subplot(act_id, [subj_id], pipeline='without_med')
                if fig_avg_bw:
                    avg_file_bw = os.path.join(subject_dir, f"S{subj_id:02d}_average_without_med.png")
                    fig_avg_bw.savefig(avg_file_bw, dpi=150)
                    plt.close(fig_avg_bw)
                    print(f"[INFO] Saved {avg_file_bw}")

    # ---------------- HELPER METHODS ---------------- #
    def _compute_label(self, trial) -> int:
        """Basic logic for 'fd' => 1 if action_id>9 else 0, or 'age', etc."""
        if self.task == 'fd':
            return int(trial.action_id > 9)
        elif self.task == 'age':
            return int(trial.subject_id < 29 or trial.subject_id > 46)
        else:
            return trial.action_id - 1

    def _copy_to_needs_cleaning(self, needs_cleaning_dir: str, file_path: str):
        import shutil
        try:
            rel_path = os.path.relpath(os.path.dirname(file_path), self.dataset.root_dir)
            target_dir = os.path.join(needs_cleaning_dir, rel_path)
            os.makedirs(target_dir, exist_ok=True)
            out_file = os.path.join(target_dir, os.path.basename(file_path))
            shutil.copy2(file_path, out_file)
            print(f"[INFO] Successfully copied {os.path.basename(file_path)} to {out_file} for cleaning")
        except Exception as e:
            print(f"[ERROR] Error copying file {file_path}: {e}")

    # ---------------- VISUALIZATION METHODS ---------------- #
    def visualize_subject_activity(
        self, 
        relative_idx_list: List[int], 
        subject_id: int, 
        activity_id: int, 
        pipeline: str = 'with_med'
    ) -> Optional[plt.Figure]:
        """
        Up to 5 trials. 6 rows => 3 axes phone/watch, 2 columns => raw vs. filter+norm
        pipeline: 'with_med' for Median + Butterworth, 'without_med' for Butterworth Only
        """
        print(f"[DEBUG] visualize_subject_activity => S{subject_id:02d}A{activity_id:02d}, Pipeline: {pipeline}")
        num_trials = min(5, len(relative_idx_list))
        rows = 6
        cols = num_trials * 2

        # Ensure axes is always 2D
        fig, axes = plt.subplots(
            nrows=rows, 
            ncols=cols, 
            figsize=(cols * 3, rows * 2.5), 
            sharex=False, 
            squeeze=False  # Ensures axes is always 2D
        )

        axis_labels = ["X", "Y", "Z"]
        phone_key = "accelerometer_phone"
        watch_key = "accelerometer_watch"

        # Select the appropriate filtered and normalized data based on pipeline
        if pipeline == 'with_med':
            filt_key = f"{phone_key}_with_med"
            filt_wt_key = f"{watch_key}_with_med"
            norm_key = f"{phone_key}_with_med"
            norm_wt_key = f"{watch_key}_with_med"
        elif pipeline == 'without_med':
            filt_key = f"{phone_key}_without_med"
            filt_wt_key = f"{watch_key}_without_med"
            norm_key = f"{phone_key}_without_med"
            norm_wt_key = f"{watch_key}_without_med"
        else:
            print(f"[ERROR] Unknown pipeline: {pipeline}")
            return None

        for t_i, relative_idx in enumerate(relative_idx_list[:5]):
            c0, c1 = t_i * 2, t_i * 2 + 1

            # Retrieve global trial index from relative index
            global_idx = (
                self.successful_indices_with_med[relative_idx] 
                if pipeline == 'with_med' 
                else self.successful_indices_without_med[relative_idx]
            )

            ph_raw = (
                self._get_raw_data_with_med(phone_key, relative_idx) 
                if pipeline == 'with_med' 
                else self._get_raw_data_without_med(phone_key, relative_idx)
            )
            wt_raw = (
                self._get_raw_data_with_med(watch_key, relative_idx) 
                if pipeline == 'with_med' 
                else self._get_raw_data_without_med(watch_key, relative_idx)
            )

            # Retrieve filtered and normalized data based on pipeline
            ph_filt = (
                self._get_filtered_with_med_data(phone_key, global_idx) 
                if pipeline == 'with_med' 
                else self._get_filtered_without_med_data(phone_key, global_idx)
            )
            wt_filt = (
                self._get_filtered_with_med_data(watch_key, global_idx) 
                if pipeline == 'with_med' 
                else self._get_filtered_without_med_data(watch_key, global_idx)
            )
            ph_norm = (
                self._get_norm_with_med_data(phone_key, global_idx) 
                if pipeline == 'with_med' 
                else self._get_norm_without_med_data(phone_key, global_idx)
            )
            wt_norm = (
                self._get_norm_with_med_data(watch_key, global_idx) 
                if pipeline == 'with_med' 
                else self._get_norm_without_med_data(watch_key, global_idx)
            )

            for axis_i in range(3):
                rp = axis_i * 2
                rw = rp + 1

                # Raw Column
                ax_ph_raw = axes[rp, c0]
                ax_wt_raw = axes[rw, c0]
                if ph_raw is not None:
                    ax_ph_raw.plot(ph_raw[:, axis_i], color='blue', label="Ph-Raw")
                if wt_raw is not None:
                    ax_wt_raw.plot(wt_raw[:, axis_i], color='green', label="Wt-Raw")

                if axis_i == 0:
                    ax_ph_raw.set_title(f"T{t_i+1:02d}\nRAW")
                    ax_wt_raw.set_title(f"T{t_i+1:02d}\nRAW")

                ax_ph_raw.set_ylabel(f"{axis_labels[axis_i]}(Ph)", fontsize=8)
                ax_wt_raw.set_ylabel(f"{axis_labels[axis_i]}(Wt)", fontsize=8)
                ax_ph_raw.grid(True)
                ax_wt_raw.grid(True)

                # Legend
                ph_handles, ph_labels = ax_ph_raw.get_legend_handles_labels()
                if ph_labels:
                    ax_ph_raw.legend(fontsize=7, loc='upper right')
                wt_handles, wt_labels = ax_wt_raw.get_legend_handles_labels()
                if wt_labels:
                    ax_wt_raw.legend(fontsize=7, loc='upper right')

                # Filter + Norm Column
                ax_ph_fln = axes[rp, c1]
                ax_wt_fln = axes[rw, c1]
                if ph_filt is not None:
                    ax_ph_fln.plot(ph_filt[:, axis_i], color='blue', label="Ph-Filt")
                if ph_norm is not None:
                    ax_ph_fln.plot(ph_norm[:, axis_i], color='red', linestyle='--', label="Ph-Norm")

                if wt_filt is not None:
                    ax_wt_fln.plot(wt_filt[:, axis_i], color='green', label="Wt-Filt")
                if wt_norm is not None:
                    ax_wt_fln.plot(wt_norm[:, axis_i], color='orange', linestyle='--', label="Wt-Norm")

                if axis_i == 0:
                    ax_ph_fln.set_title(f"T{t_i+1:02d}\nFilt+Norm")
                    ax_wt_fln.set_title(f"T{t_i+1:02d}\nFilt+Norm")

                ax_ph_fln.set_ylabel(f"{axis_labels[axis_i]}(Ph)", fontsize=8)
                ax_wt_fln.set_ylabel(f"{axis_labels[axis_i]}(Wt)", fontsize=8)
                ax_ph_fln.grid(True)
                ax_wt_fln.grid(True)

                # Legend
                ph_fln_handles, ph_fln_labels = ax_ph_fln.get_legend_handles_labels()
                if ph_fln_labels:
                    ax_ph_fln.legend(fontsize=7, loc='upper right')
                wt_fln_handles, wt_fln_labels = ax_wt_fln.get_legend_handles_labels()
                if wt_fln_labels:
                    ax_wt_fln.legend(fontsize=7, loc='upper right')

        pipeline_title = "Median + Butterworth" if pipeline == 'with_med' else "Butterworth Only"
        plt.suptitle(
            f"Subject {subject_id:02d}, Activity {activity_id:02d}\nRaw vs. Filter+Norm ({pipeline_title})", 
            fontsize=10
        )
        plt.tight_layout()
        return fig

    def visualize_subject_activity_normonly(
        self, 
        relative_idx_list: List[int], 
        subject_id: int, 
        activity_id: int, 
        pipeline: str = 'with_med'
    ) -> Optional[plt.Figure]:
        """
        Norm-only phone/watch. 3 axes => X,Y,Z. Each axis => phone row + watch row => 6 rows total.
        Up to 5 columns => T01..T05
        pipeline: 'with_med' for Median + Butterworth, 'without_med' for Butterworth Only
        """
        print(f"[DEBUG] visualize_subject_activity_normonly => S{subject_id:02d}A{activity_id:02d}, Pipeline: {pipeline}")
        num_trials = min(5, len(relative_idx_list))
        rows = 6  # 3 axes => each axis has phone row + watch row
        cols = num_trials
        axis_labels = ['X', 'Y', 'Z']

        # Ensure axes is always 2D
        fig, axes = plt.subplots(
            nrows=rows, 
            ncols=cols, 
            figsize=(cols * 2.5, rows * 2), 
            sharex=False, 
            squeeze=False  # Ensures axes is always 2D
        )

        phone_key = 'accelerometer_phone'
        watch_key = 'accelerometer_watch'

        # Select the appropriate normalized data based on pipeline
        if pipeline == 'with_med':
            norm_ph_key = f"{phone_key}_with_med"
            norm_wt_key = f"{watch_key}_with_med"
        elif pipeline == 'without_med':
            norm_ph_key = f"{phone_key}_without_med"
            norm_wt_key = f"{watch_key}_without_med"
        else:
            print(f"[ERROR] Unknown pipeline: {pipeline}")
            return None

        for t_i, relative_idx in enumerate(relative_idx_list[:5]):
            # Retrieve global trial index from relative index
            global_idx = (
                self.successful_indices_with_med[relative_idx] 
                if pipeline == 'with_med' 
                else self.successful_indices_without_med[relative_idx]
            )

            # Retrieve normalized data based on pipeline
            ph_norm = (
                self._get_norm_with_med_data(phone_key, global_idx) 
                if pipeline == 'with_med' 
                else self._get_norm_without_med_data(phone_key, global_idx)
            )
            wt_norm = (
                self._get_norm_with_med_data(watch_key, global_idx) 
                if pipeline == 'with_med' 
                else self._get_norm_without_med_data(watch_key, global_idx)
            )

            for axis_i in range(3):
                rp = axis_i * 2
                rw = rp + 1

                ax_ph = axes[rp, t_i]
                ax_wt = axes[rw, t_i]

                if ph_norm is not None:
                    ax_ph.plot(ph_norm[:, axis_i], color='blue', label="Ph-Norm")
                if wt_norm is not None:
                    ax_wt.plot(wt_norm[:, axis_i], color='green', label="Wt-Norm")

                if axis_i == 0:
                    ax_ph.set_title(f"T{t_i+1:02d}\n(Phone)", fontsize=8)
                    ax_wt.set_title(f"T{t_i+1:02d}\n(Watch)", fontsize=8)
                ax_ph.set_ylabel(f"{axis_labels[axis_i]}(Ph)", fontsize=8)
                ax_wt.set_ylabel(f"{axis_labels[axis_i]}(Wt)", fontsize=8)
                ax_ph.grid(True, linestyle='--', alpha=0.5)
                ax_wt.grid(True, linestyle='--', alpha=0.5)

                # Legend
                ph_handles, ph_labels = ax_ph.get_legend_handles_labels()
                if ph_labels:
                    ax_ph.legend(fontsize=6, loc='upper right')
                wt_handles, wt_labels = ax_wt.get_legend_handles_labels()
                if wt_labels:
                    ax_wt.legend(fontsize=6, loc='upper right')

        pipeline_title = "Median + Butterworth" if pipeline == 'with_med' else "Butterworth Only"
        plt.suptitle(
            f"Subject {subject_id:02d}, Activity {activity_id:02d}\nNorm-only (Phone vs Watch) ({pipeline_title})", 
            fontsize=10
        )
        plt.tight_layout()
        return fig

    def visualize_activity_average(
        self, 
        activity_id: int, 
        pipeline: str = 'with_med'
    ) -> Optional[plt.Figure]:
        """
        Visualizes the per-activity average filtered and normalized data across all subjects.

        Parameters:
        - activity_id: ID of the activity.
        - pipeline: 'with_med' for Median + Butterworth, 'without_med' for Butterworth Only.

        Returns:
        - Matplotlib Figure object or None if no data available.
        """
        print(f"[DEBUG] visualize_activity_average => Activity={activity_id}, Pipeline: {pipeline}")
        if pipeline == 'with_med':
            filt_keys = [k for k in self.data_with_med.keys() if k.startswith('accelerometer_') and not k.endswith('_labels')]
            norm_keys = list(self._normalized_with_med.keys())
        elif pipeline == 'without_med':
            filt_keys = [k for k in self.data_without_med.keys() if k.startswith('accelerometer_') and not k.endswith('_labels')]
            norm_keys = list(self._normalized_without_med.keys())
        else:
            print(f"[ERROR] Unknown pipeline: {pipeline}")
            return None

        if not filt_keys:
            print(f"[WARNING] No filtered data found for Activity {activity_id} and Pipeline {pipeline}")
            return None

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 9), sharex=True)
        axis_labels = ['X', 'Y', 'Z']

        for axis_i in range(3):
            for sensor_idx, sensor_key in enumerate(['accelerometer_phone', 'accelerometer_watch']):
                filt_key = f"{sensor_key}_with_med" if pipeline == 'with_med' else f"{sensor_key}_without_med"
                norm_key = f"{sensor_key}_with_med" if pipeline == 'with_med' else f"{sensor_key}_without_med"

                # Check if the key exists in the respective dataset
                if pipeline == 'with_med' and filt_key not in self.data_with_med:
                    print(f"[WARNING] Missing data for {filt_key}")
                    continue
                if pipeline == 'without_med' and filt_key not in self.data_without_med:
                    print(f"[WARNING] Missing data for {filt_key}")
                    continue

                # Filtered average
                filt_data = self.data_with_med[filt_key] if pipeline == 'with_med' else self.data_without_med[filt_key]
                filt_avg = np.mean(filt_data[:, axis_i, :], axis=0) if filt_data.size > 0 else None

                # Normalized average
                norm_data = self._normalized_with_med.get(norm_key, None) if pipeline == 'with_med' else self._normalized_without_med.get(norm_key, None)
                norm_avg = np.mean(norm_data[:, axis_i, :], axis=0) if norm_data is not None and norm_data.size > 0 else None

                ax = axes[axis_i, sensor_idx]
                if filt_avg is not None:
                    ax.plot(filt_avg, color='blue', label='Filt-Avg')
                if norm_avg is not None:
                    ax.plot(norm_avg, color='red', linestyle='--', label='Norm-Avg')

                ax.set_title(f"{sensor_key.replace('accelerometer_', '').capitalize()} - {axis_labels[axis_i]} Axis")
                ax.set_ylabel(f"{axis_labels[axis_i]} Value")
                ax.grid(True, linestyle='--', alpha=0.5)

                if axis_i == 0 and sensor_idx == 1:
                    ax.legend(fontsize=8, loc='upper right')

        pipeline_title = "Median + Butterworth" if pipeline == 'with_med' else "Butterworth Only"
        plt.suptitle(f"Activity {activity_id:02d} Average\n({pipeline_title} Pipeline)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def create_all_activities_summary(
        self, 
        activity_list: List[int], 
        pipeline: str = 'with_med'
    ) -> Optional[plt.Figure]:
        """
        Combines multiple activities into one figure, showing averages for each activity.

        Parameters:
        - activity_list: List of activity IDs to include.
        - pipeline: 'with_med' for Median + Butterworth, 'without_med' for Butterworth Only.

        Returns:
        - Matplotlib Figure object or None if no data available.
        """
        print(f"[DEBUG] create_all_activities_summary => Activities={activity_list}, Pipeline: {pipeline}")
        num_activities = len(activity_list)
        if num_activities == 0:
            print("[WARNING] No activities provided for summary.")
            return None

        fig, axes = plt.subplots(nrows=num_activities, ncols=1, figsize=(14, num_activities * 4), sharex=True)
        if num_activities == 1:
            axes = [axes]

        for a_i, activity_id in enumerate(activity_list):
            avg_fig = self.visualize_activity_average(activity_id, pipeline=pipeline)
            if avg_fig:
                # Since visualize_activity_average creates its own figure, we'll not embed it.
                # Instead, we can add a placeholder or refactor visualize_activity_average to return data.
                plt.close(avg_fig)  # Close the individual figure
                axes[a_i].text(
                    0.5, 0.5, 
                    f"Activity {activity_id}: Average Plot\n({pipeline})", 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    fontsize=12
                )
                axes[a_i].axis('off')

        pipeline_title = "Median + Butterworth" if pipeline == 'with_med' else "Butterworth Only"
        plt.suptitle(f"All Activities Summary\n({pipeline_title} Pipeline)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def visualize_activity_subject_averages_in_same_subplot(
        self, 
        activity_id: int, 
        subject_list: List[int], 
        pipeline: str = 'with_med'
    ) -> Optional[plt.Figure]:
        """
        Each subject => 1 row, 3 columns => X, Y, Z.
        Overlaid lines => phone_filt, phone_norm, watch_filt, watch_norm (averaged across all trials).
        pipeline: 'with_med' for Median + Butterworth, 'without_med' for Butterworth Only
        """
        print(f"[DEBUG] visualize_activity_subject_averages_in_same_subplot => Activity={activity_id}, subjects={subject_list}, Pipeline: {pipeline}")
        if not subject_list:
            print("[WARNING] No subjects provided for averaging visualization.")
            return None

        rows = len(subject_list)
        cols = 3
        fig, axes = plt.subplots(
            nrows=rows, 
            ncols=cols, 
            figsize=(cols * 4, rows * 3), 
            sharex=False, 
            squeeze=False  # Ensures axes is always 2D
        )

        axis_names = ['X', 'Y', 'Z']
        phone_key = "accelerometer_phone"
        watch_key = "accelerometer_watch"

        # Select the appropriate normalized data based on pipeline
        if pipeline == 'with_med':
            filt_keys = [f"{phone_key}_with_med", f"{watch_key}_with_med"]
            norm_keys = [f"{phone_key}_with_med", f"{watch_key}_with_med"]
        elif pipeline == 'without_med':
            filt_keys = [f"{phone_key}_without_med", f"{watch_key}_without_med"]
            norm_keys = [f"{phone_key}_without_med", f"{watch_key}_without_med"]
        else:
            print(f"[ERROR] Unknown pipeline: {pipeline}")
            return None

        for s_i, subj_id in enumerate(subject_list):
            ph_filt_avg, ph_norm_avg = self._compute_subject_activity_average(
                subj_id, activity_id, phone_key, pipeline=pipeline
            )
            wt_filt_avg, wt_norm_avg = self._compute_subject_activity_average(
                subj_id, activity_id, watch_key, pipeline=pipeline
            )

            for axis_i in range(3):
                ax = axes[s_i, axis_i]  # Consistently index axes as 2D

                # Overlaid lines
                if ph_filt_avg is not None:
                    ax.plot(ph_filt_avg[:, axis_i], color='blue', label='Ph-Filt')
                if ph_norm_avg is not None:
                    ax.plot(ph_norm_avg[:, axis_i], color='red', linestyle='--', label='Ph-Norm')
                if wt_filt_avg is not None:
                    ax.plot(wt_filt_avg[:, axis_i], color='green', label='Wt-Filt')
                if wt_norm_avg is not None:
                    ax.plot(wt_norm_avg[:, axis_i], color='orange', linestyle='--', label='Wt-Norm')

                ax.set_title(f"S{subj_id:02d}-{axis_names[axis_i]}", fontsize=10)
                ax.set_ylabel(f"{axis_names[axis_i]} Value")
                ax.grid(True, linestyle='--', alpha=0.5)

                # Show legend only for the first subplot
                if s_i == 0 and axis_i == 2 and (
                    ph_filt_avg is not None or 
                    ph_norm_avg is not None or 
                    wt_filt_avg is not None or 
                    wt_norm_avg is not None
                ):
                    ax.legend(fontsize=6, loc='upper right')

        pipeline_title = "Median + Butterworth" if pipeline == 'with_med' else "Butterworth Only"
        plt.suptitle(
            f"Activity {activity_id:02d}: Subject-Average Filter+Norm (Same Subplot) ({pipeline_title})", 
            fontsize=16
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def visualize_activity_all_trials_same_subplot(
        self, 
        activity_id: int, 
        subject_list: List[int], 
        pipeline: str = 'with_med'
    ) -> Optional[plt.Figure]:
        """
        Visualizes all trials for each subject in the same subplot.
        Each row represents a subject, and each column represents a trial.
        Overlays phone and watch filtered + normalized data within each trial subplot.

        Parameters:
        - activity_id: ID of the activity.
        - subject_list: List of subject IDs to include.
        - pipeline: 'with_med' for Median + Butterworth, 'without_med' for Butterworth Only.

        Returns:
        - Matplotlib Figure object or None if no data available.
        """
        print(f"[DEBUG] visualize_activity_all_trials_same_subplot => Activity={activity_id}, subjects={subject_list}, Pipeline: {pipeline}")
        if not subject_list:
            print("[WARNING] No subjects provided for trials visualization.")
            return None

        n_subj = len(subject_list)
        cols = 5  # Up to 5 trials per subject
        fig, axes = plt.subplots(
            nrows=n_subj, 
            ncols=cols, 
            figsize=(cols * 3, n_subj * 3), 
            sharex=False, 
            squeeze=False  # Ensures axes is always 2D
        )

        phone_key = 'accelerometer_phone'
        watch_key = 'accelerometer_watch'

        # Select the appropriate filtered and normalized data based on pipeline
        if pipeline == 'with_med':
            filt_keys = [f"{phone_key}_with_med", f"{watch_key}_with_med"]
            norm_keys = [f"{phone_key}_with_med", f"{watch_key}_with_med"]
        elif pipeline == 'without_med':
            filt_keys = [f"{phone_key}_without_med", f"{watch_key}_without_med"]
            norm_keys = [f"{phone_key}_without_med", f"{watch_key}_without_med"]
        else:
            print(f"[ERROR] Unknown pipeline: {pipeline}")
            return None

        for s_i, subj_id in enumerate(subject_list):
            # Find all trials for (subj_id, activity_id)
            trial_indexes = []
            for relative_idx, global_idx in enumerate(
                self.successful_indices_with_med if pipeline == 'with_med' else self.successful_indices_without_med
            ):
                trial = self.dataset.matched_trials[global_idx]
                if trial.subject_id == subj_id and trial.action_id == activity_id:
                    trial_indexes.append((trial.sequence_number, relative_idx))
            trial_indexes.sort(key=lambda x: x[0])

            for c_i in range(cols):
                ax = axes[s_i, c_i]

                if c_i < len(trial_indexes):
                    seq_num, relative_idx = trial_indexes[c_i]
                    global_idx = (
                        self.successful_indices_with_med[relative_idx] 
                        if pipeline == 'with_med' 
                        else self.successful_indices_without_med[relative_idx]
                    )
                    ph_filt = (
                        self._get_filtered_with_med_data(phone_key, global_idx) 
                        if pipeline == 'with_med' 
                        else self._get_filtered_without_med_data(phone_key, global_idx)
                    )
                    ph_norm = (
                        self._get_norm_with_med_data(phone_key, global_idx) 
                        if pipeline == 'with_med' 
                        else self._get_norm_without_med_data(phone_key, global_idx)
                    )
                    wt_filt = (
                        self._get_filtered_with_med_data(watch_key, global_idx) 
                        if pipeline == 'with_med' 
                        else self._get_filtered_without_med_data(watch_key, global_idx)
                    )
                    wt_norm = (
                        self._get_norm_with_med_data(watch_key, global_idx) 
                        if pipeline == 'with_med' 
                        else self._get_norm_without_med_data(watch_key, global_idx)
                    )

                    # Overlay all 3 axes with different colors and line styles
                    axis_colors = ['blue', 'red', 'green']
                    for axis_i in range(3):
                        if ph_filt is not None:
                            ax.plot(ph_filt[:, axis_i], color=axis_colors[axis_i],
                                    label='Ph-Filt' if (axis_i == 0 and c_i == 0 and s_i == 0) else None)
                        if ph_norm is not None:
                            ax.plot(ph_norm[:, axis_i], color=axis_colors[axis_i], linestyle='--',
                                    label='Ph-Norm' if (axis_i == 0 and c_i == 0 and s_i == 0) else None)
                        if wt_filt is not None:
                            ax.plot(wt_filt[:, axis_i], color=axis_colors[axis_i], alpha=0.5,
                                    label='Wt-Filt' if (axis_i == 0 and c_i == 0 and s_i == 0) else None)
                        if wt_norm is not None:
                            ax.plot(wt_norm[:, axis_i], color=axis_colors[axis_i], linestyle='--', alpha=0.5,
                                    label='Wt-Norm' if (axis_i == 0 and c_i == 0 and s_i == 0) else None)

                    ax.set_title(f"S{subj_id:02d}T{seq_num:02d}", fontsize=10)
                    ax.grid(True, linestyle='--', alpha=0.5)
                    # Show legend only in the first subplot
                    if s_i == 0 and c_i == 0:
                        handles, labels_ = ax.get_legend_handles_labels()
                        if labels_:
                            ax.legend(fontsize=6, loc='upper right')
                else:
                    ax.set_facecolor('#f9f9f9')
                    ax.set_title("NoTrial", fontsize=8)
                    ax.axis('off')  # Hide axes for empty subplots

        pipeline_title = "Median + Butterworth" if pipeline == 'with_med' else "Butterworth Only"
        plt.suptitle(
            f"Activity {activity_id:02d}: All Trials Filter+Norm in Same Subplot\n(Phone+Watch Overlaid) ({pipeline_title})", 
            fontsize=16
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def visualize_trial_normxyz_smv(
        self, 
        matched_idx: int, 
        pipeline: str = 'with_med'
    ) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
        """
        Produces two figures:
          1) fig_xyz: Separate subplots for X, Y, Z, each overlaying phone vs watch for that axis (normalized)
          2) fig_smv: Overlays phone vs watch for SMV (the 4th column)

        Parameters:
        - matched_idx: index in self.dataset.matched_trials
        - pipeline: 'with_med' for Median + Butterworth, 'without_med' for Butterworth Only

        Returns:
            (fig_xyz, fig_smv)
            or (None, None) if data not found.
        """
        phone_key = "accelerometer_phone"
        watch_key = "accelerometer_watch"

        # Select the appropriate normalized data based on pipeline
        if pipeline == 'with_med':
            ph_norm = self._get_norm_with_med_data(phone_key, matched_idx)
            wt_norm = self._get_norm_with_med_data(watch_key, matched_idx)
        elif pipeline == 'without_med':
            ph_norm = self._get_norm_without_med_data(phone_key, matched_idx)
            wt_norm = self._get_norm_without_med_data(watch_key, matched_idx)
        else:
            print(f"[ERROR] Unknown pipeline: {pipeline}")
            return (None, None)

        if ph_norm is None or ph_norm.shape[1] < 3:
            print(f"[ERROR] Trial index {matched_idx} missing phone normalized data or insufficient columns.")
            return (None, None)
        if wt_norm is None or wt_norm.shape[1] < 3:
            print(f"[ERROR] Trial index {matched_idx} missing watch normalized data or insufficient columns.")
            return (None, None)

        # Calculate SMV if not present (assuming SMV is not precomputed)
        # SMV = sqrt(X^2 + Y^2 + Z^2)
        if ph_norm.shape[1] < 4:
            ph_smv = np.sqrt(np.sum(ph_norm[:, :3] ** 2, axis=1))
            ph_norm = np.hstack((ph_norm, ph_smv.reshape(-1, 1)))
        if wt_norm.shape[1] < 4:
            wt_smv = np.sqrt(np.sum(wt_norm[:, :3] ** 2, axis=1))
            wt_norm = np.hstack((wt_norm, wt_smv.reshape(-1, 1)))

        # 1) Create fig_xyz with separate subplots for X, Y, Z
        fig_xyz, axes_xyz = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
        axis_labels = ['X', 'Y', 'Z']
        time_axis = np.arange(len(ph_norm)) * (1 / self.fs)  # Convert sample index to time in seconds

        for i, axis_label in enumerate(axis_labels):
            # Phone data
            axes_xyz[i].plot(time_axis, ph_norm[:, i], color='blue', label='Phone')
            # Watch data
            axes_xyz[i].plot(time_axis, wt_norm[:, i], color='green', linestyle='--', label='Watch')
            axes_xyz[i].set_ylabel(f"{axis_label}-Axis")
            axes_xyz[i].grid(True, linestyle='--', alpha=0.5)
            if i == 0:
                axes_xyz[i].legend(fontsize=8, loc='upper right')

        axes_xyz[2].set_xlabel("Time (seconds)")
        pipeline_title = "Median + Butterworth" if pipeline == 'with_med' else "Butterworth Only"
        fig_xyz.suptitle(f"Normalized X, Y, Z (Phone vs Watch) ({pipeline_title})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 2) Create fig_smv for SMV
        fig_smv, ax_smv = plt.subplots(figsize=(12, 6))
        # Phone SMV
        ax_smv.plot(time_axis, ph_norm[:, 3], color='blue', label='Phone-SMV')
        # Watch SMV
        ax_smv.plot(time_axis, wt_norm[:, 3], color='orange', linestyle='--', label='Watch-SMV')
        ax_smv.set_title(f"Normalized SMV (Phone vs Watch) ({pipeline_title})")
        ax_smv.set_xlabel("Time (seconds)")
        ax_smv.set_ylabel("SMV")
        ax_smv.grid(True, linestyle='--', alpha=0.5)
        ax_smv.legend(fontsize=8, loc='upper right')

        plt.tight_layout()

        return (fig_xyz, fig_smv)

    def visualize_trial_comparison(self, matched_idx: int) -> Optional[plt.Figure]:
        """
        Produces a figure comparing:
            - Butterworth Only Filtered Data
            - Median + Butterworth Filtered Data
        for each axis (X, Y, Z) and SMV in separate subplots for phone and watch.

        Parameters:
        - matched_idx: index in self.dataset.matched_trials

        Returns:
            fig: Matplotlib Figure object or None if data is missing
        """
        phone_key = "accelerometer_phone"
        watch_key = "accelerometer_watch"

        # Retrieve normalized data for both pipelines
        ph_norm_with_med = self._get_norm_with_med_data(phone_key, matched_idx)
        wt_norm_with_med = self._get_norm_with_med_data(watch_key, matched_idx)
        ph_norm_without_med = self._get_norm_without_med_data(phone_key, matched_idx)
        wt_norm_without_med = self._get_norm_without_med_data(watch_key, matched_idx)

        # Check data availability
        if (ph_norm_with_med is None or wt_norm_with_med is None or
            ph_norm_without_med is None or wt_norm_without_med is None):
            print(f"[ERROR] Trial index {matched_idx} missing normalized data for comparison.")
            return None

        # Calculate SMV if not present (assuming SMV is not precomputed)
        if ph_norm_with_med.shape[1] < 4:
            ph_smv_with_med = np.sqrt(np.sum(ph_norm_with_med[:, :3] ** 2, axis=1))
            ph_norm_with_med = np.hstack((ph_norm_with_med, ph_smv_with_med.reshape(-1, 1)))
        if wt_norm_with_med.shape[1] < 4:
            wt_smv_with_med = np.sqrt(np.sum(wt_norm_with_med[:, :3] ** 2, axis=1))
            wt_norm_with_med = np.hstack((wt_norm_with_med, wt_smv_with_med.reshape(-1, 1)))
        if ph_norm_without_med.shape[1] < 4:
            ph_smv_without_med = np.sqrt(np.sum(ph_norm_without_med[:, :3] ** 2, axis=1))
            ph_norm_without_med = np.hstack((ph_norm_without_med, ph_smv_without_med.reshape(-1, 1)))
        if wt_norm_without_med.shape[1] < 4:
            wt_smv_without_med = np.sqrt(np.sum(wt_norm_without_med[:, :3] ** 2, axis=1))
            wt_norm_without_med = np.hstack((wt_norm_without_med, wt_smv_without_med.reshape(-1, 1)))

        # Create figure with 4 subplots: Phone X, Y, Z, SMV and Watch X, Y, Z, SMV
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 12), sharex=True)
        axis_labels = ['X', 'Y', 'Z', 'SMV']
        sensors = ['Phone', 'Watch']

        for sensor_idx, sensor_key in enumerate([phone_key, watch_key]):
            if sensor_key == phone_key:
                norm_with_med = ph_norm_with_med
                norm_without_med = ph_norm_without_med
            else:
                norm_with_med = wt_norm_with_med
                norm_without_med = wt_norm_without_med

            for i, axis_label in enumerate(axis_labels):
                ax = axes[i, sensor_idx]
                if i < 3:
                    # X, Y, Z axes
                    ax.plot(norm_with_med[:, i], color='blue', label='Median + Butterworth')
                    ax.plot(norm_without_med[:, i], color='red', linestyle='--', label='Butterworth Only')
                    ax.set_ylabel(f"{axis_label}-Axis")
                else:
                    # SMV
                    ax.plot(norm_with_med[:, i], color='blue', label='Median + Butterworth')
                    ax.plot(norm_without_med[:, i], color='red', linestyle='--', label='Butterworth Only')

                ax.set_title(f"{sensors[sensor_idx]} - {axis_label}")
                ax.grid(True, linestyle='--', alpha=0.5)

                if i == 0:
                    ax.legend(fontsize=8, loc='upper right')

        axes[3, 0].set_xlabel("Time (seconds)")
        axes[3, 1].set_xlabel("Time (seconds)")
        fig.suptitle("Comparison of Filtering Pipelines: Median + Butterworth vs. Butterworth Only", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig

    # ---------------- HELPER DATA RETRIEVAL METHODS ---------------- #
    def _get_raw_data_with_med(self, key: str, relative_idx: int) -> Optional[np.ndarray]:
        if key in self._raw_data_with_med and relative_idx < len(self._raw_data_with_med[key]):
            return self._raw_data_with_med[key][relative_idx]
        return None

    def _get_raw_data_without_med(self, key: str, relative_idx: int) -> Optional[np.ndarray]:
        if key in self._raw_data_without_med and relative_idx < len(self._raw_data_without_med[key]):
            return self._raw_data_without_med[key][relative_idx]
        return None

    def _get_filtered_with_med_data(self, key: str, global_idx: int) -> Optional[np.ndarray]:
        if key in self._filtered_with_med:
            try:
                relative_idx = self.successful_indices_with_med.index(global_idx)
                return self._filtered_with_med[key][relative_idx]
            except ValueError:
                return None
        return None

    def _get_filtered_without_med_data(self, key: str, global_idx: int) -> Optional[np.ndarray]:
        if key in self._filtered_without_med:
            try:
                relative_idx = self.successful_indices_without_med.index(global_idx)
                return self._filtered_without_med[key][relative_idx]
            except ValueError:
                return None
        return None

    def _get_norm_with_med_data(self, key: str, global_idx: int) -> Optional[np.ndarray]:
        if key in self._normalized_with_med:
            try:
                relative_idx = self.successful_indices_with_med.index(global_idx)
                if relative_idx < self._normalized_with_med[key].shape[0]:
                    return self._normalized_with_med[key][relative_idx]
            except ValueError:
                return None
        return None

    def _get_norm_without_med_data(self, key: str, global_idx: int) -> Optional[np.ndarray]:
        if key in self._normalized_without_med:
            try:
                relative_idx = self.successful_indices_without_med.index(global_idx)
                if relative_idx < self._normalized_without_med[key].shape[0]:
                    return self._normalized_without_med[key][relative_idx]
            except ValueError:
                return None
        return None

    def _compute_subject_activity_average(
        self, 
        subject_id: int, 
        activity_id: int, 
        modality_key: str, 
        pipeline: str = 'with_med'
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Computes the average filtered and normalized data across all trials for a given subject and activity.

        Parameters:
        - subject_id: ID of the subject
        - activity_id: ID of the activity
        - modality_key: Modality key (e.g., 'accelerometer_phone')
        - pipeline: 'with_med' for Median + Butterworth, 'without_med' for Butterworth Only

        Returns:
            (filtered_average, normalized_average)
        """
        if pipeline == 'with_med':
            filt_key = f"{modality_key}_with_med"
            norm_key = f"{modality_key}_with_med"
            norm_data = self._normalized_with_med.get(norm_key, None)
            filt_data = self.data_with_med.get(filt_key, [])
        elif pipeline == 'without_med':
            filt_key = f"{modality_key}_without_med"
            norm_key = f"{modality_key}_without_med"
            norm_data = self._normalized_without_med.get(norm_key, None)
            filt_data = self.data_without_med.get(filt_key, [])
        else:
            print(f"[ERROR] Unknown pipeline: {pipeline}")
            return (None, None)

        if filt_data and norm_data is not None:
            filt_avg = np.mean(filt_data, axis=0)
            norm_avg = np.mean(norm_data, axis=0)
            return (filt_avg, norm_avg)
        else:
            return (None, None)