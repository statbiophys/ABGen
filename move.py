import os
import shutil

def copy_files_with_string(folder_path, include_string, exclude_string, destination_folder):
    # Check if the provided folder paths are valid
    if not os.path.isdir(folder_path):
        print(f"The path '{folder_path}' is not a valid directory.")
        return
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)  # Create the destination folder if it doesn't exist

    # Search and copy files that contain include_string but not exclude_string in their names
    files_copied = 0
    for filename in os.listdir(folder_path):
        if include_string in filename and exclude_string not in filename:
            source_file = os.path.join(folder_path, filename)
            destination_file = os.path.join(destination_folder, filename)
            if os.path.isfile(source_file):
                shutil.copy2(source_file, destination_file)
                files_copied += 1
                print(f"Copied: {filename}")

    if files_copied == 0:
        print("No files found matching the specified criteria.")
    else:
        print(f"{files_copied} file(s) copied to '{destination_folder}'.")


# Example usage
folder_path = './lib/dataset'
include_string = 'preprocess'
exclude_string = 'monster'
destination_folder = './lib/dataset/preprocess'
copy_files_with_string(folder_path, include_string, exclude_string, destination_folder)
