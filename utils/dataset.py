import os
import re
from typing import List, Dict, Optional
import numpy as np

# ---------------------- DATA CLASSES ---------------------- #

class ModalityFile:
    def __init__(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

    def __repr__(self) -> str:
        return (f"ModalityFile(subject_id={self.subject_id}, action_id={self.action_id}, "
                f"sequence_number={self.sequence_number}, file_path='{self.file_path}')")


class MatchedTrial:
    id_counter = 0  # Class-level counter for unique IDs

    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}
        self.id = MatchedTrial.id_counter  # Assign unique ID
        MatchedTrial.id_counter += 1       # Increment counter

    def add_file(self, modality_sensor_key: str, file_path: str) -> None:
        self.files[modality_sensor_key] = file_path

    def __repr__(self) -> str:
        return (f"MatchedTrial(id={self.id}, subject_id={self.subject_id}, "
                f"action_id={self.action_id}, sequence_number={self.sequence_number}, files={self.files})")

class SmartFallMM:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.matched_trials: List[MatchedTrial] = []
        self.modality_sensors: Dict[str, List[str]] = {}
        self.sampling_rate = 31.25

    def add_modality(self, age_group: str, modality_name: str, sensors: Optional[List[str]] = None) -> None:
        modality_key = f"{age_group}_{modality_name}"
        self.modality_sensors[modality_key] = sensors if sensors else [None]
        print(f"Added modality {modality_key} with sensors {sensors}")

    def load_files(self) -> None:
        needs_cleaning_dir = os.path.join(self.root_dir, 'needs_cleaning')
        os.makedirs(needs_cleaning_dir, exist_ok=True)
        print(f"Using needs_cleaning directory: {needs_cleaning_dir}")

        for modality_key, sensors in self.modality_sensors.items():
            age_group, modality_name = modality_key.split('_', 1)

            for sensor in sensors:
                data_path = os.path.join(
                    self.root_dir, age_group, modality_name, sensor or ""
                )
                print(f"Looking for files in: {data_path}")
                if not os.path.exists(data_path):
                    print(f"Warning: Path does not exist: {data_path}")
                    continue

                files_found, files_with_errors = 0, 0
                for file in os.listdir(data_path):
                    if file.endswith('.csv'):
                        files_found += 1
                        file_path = os.path.join(data_path, file)
                        try:
                            # Parse filenames using regex for robust handling
                            match = re.match(r"S(\d{2})A(\d{2})T(\d{2})\.csv", file)
                            if not match:
                                raise ValueError(f"Invalid filename format: {file}")
                            subject_id, action_id, sequence_number = map(int, match.groups())

                            trial = self._find_or_create_matched_trial(
                                subject_id, action_id, sequence_number
                            )
                            key_ = f"{modality_name}_{sensor}" if sensor else modality_name
                            trial.add_file(key_, file_path)
                        except Exception as e:
                            files_with_errors += 1
                            print(f"Error processing file {file}: {e}")
                            self._handle_needs_cleaning(file, data_path, needs_cleaning_dir)
                print(f"Found {files_found} CSV files in {data_path}, {files_with_errors} errors -> cleaning dir")

    def _handle_needs_cleaning(self, file: str, src_dir: str, cleaning_dir: str) -> None:
        import shutil
        relative_path = os.path.relpath(src_dir, self.root_dir)
        target_dir = os.path.join(cleaning_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)

        src_file = os.path.join(src_dir, file)
        dest_file = os.path.join(target_dir, file)
        try:
            shutil.copy2(src_file, dest_file)
            print(f"Successfully copied {file} to {dest_file} for cleaning")
        except Exception as err:
            print(f"Error copying file {file}: {err}")

    def match_trials(self) -> None:
        required_keys = set()
        for mk, sensors in self.modality_sensors.items():
            _, modality_name = mk.split('_', 1)
            if sensors[0] is None:
                required_keys.add(modality_name)
            else:
                for sensor in sensors:
                    required_keys.add(f"{modality_name}_{sensor}")

        print(f"Before matching: {len(self.matched_trials)} trials")
        complete_trials = [
            trial for trial in self.matched_trials
            if all(k in trial.files for k in required_keys)
        ]
        print(f"After matching: {len(complete_trials)} trials (complete)")
        self.matched_trials = complete_trials

    def _find_or_create_matched_trial(self, subject_id: int, action_id: int, sequence_number: int) -> MatchedTrial:
        for trial in self.matched_trials:
            if (trial.subject_id == subject_id and
                    trial.action_id == action_id and
                    trial.sequence_number == sequence_number):
                return trial
        new_t = MatchedTrial(subject_id, action_id, sequence_number)
        self.matched_trials.append(new_t)
        return new_t

# ------------------ MAIN USAGE EXAMPLE ------------------ #

if __name__ == "__main__":
    dataset = SmartFallMM(root_dir=os.path.join(os.getcwd(), 'data', 'smartfallmm'))

    dataset.add_modality("young", "accelerometer", ["phone", "watch"])
    dataset.add_modality("young", "skeleton")


    dataset.load_files()
    dataset.match_trials()
