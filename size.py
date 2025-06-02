import os

def list_files_by_size(folder_path):
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        print(f"The path '{folder_path}' is not a valid directory.")
        return

    # Get all files in the directory along with their sizes in MB
    files_with_sizes = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Only include files, not directories
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert size to MB
            files_with_sizes.append((filename, file_size))

    # Sort files by size in descending order
    files_with_sizes.sort(key=lambda x: x[1], reverse=False)

    # Print the file names and their sizes in MB
    for filename, size in files_with_sizes:
        print(f"{filename}: {size:.2f} MB")

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size / (1024 * 1024)

# Example usage
folder_path = './SASA'
list_files_by_size(folder_path)
print(get_folder_size(folder_path))