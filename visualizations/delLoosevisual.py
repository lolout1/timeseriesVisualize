import os

# Base directory to start from
base_dir = r"/home/abheekp/Fall_Detection_KD_Multimodal/new8/timeseriesVisualize/visualizations"

# List of directories to process (A1 to A14)
directories = [f"A{i}" for i in range(1, 15)]

def delete_png_files(base_directory, sub_dirs):
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_directory, sub_dir)
        if os.path.isdir(sub_dir_path):
            print(f"Processing directory: {sub_dir_path}")
            for file_name in os.listdir(sub_dir_path):
                file_path = os.path.join(sub_dir_path, file_name)
                # Check if the file is a PNG file and not a directory
                if os.path.isfile(file_path) and file_name.endswith(".png"):
                    try:
                        os.remove(file_path)
                        print(f"Deleted file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
        else:
            print(f"Skipping {sub_dir_path}, not a directory.")

if __name__ == "__main__":
    delete_png_files(base_dir, directories)
