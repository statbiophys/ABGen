import requests
import zipfile
import os

def download_and_unzip_from_github(url, output_file, extract_to=None):
    """
    Download a file from GitHub Releases, unzip it if it's a zip file, 
    and delete the zip file after extraction.
    
    Args:
        url (str): The GitHub Releases direct download link.
        output_file (str): The local file path to save the downloaded file.
        extract_to (str): Directory where the contents of the zip file should be extracted.
                          If None, extracts in the same directory as the output_file.
    """
    print(f"Downloading from {url}...")
    try:
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Save the downloaded file
        with open(output_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"File downloaded successfully: {output_file}")

        # Check if the file is a zip file
        if zipfile.is_zipfile(output_file):
            print("The downloaded file is a zip file. Extracting...")
            
            # Set the extraction directory
            extract_to = extract_to or os.path.dirname(output_file)
            
            # Extract the zip file
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            print(f"Files extracted to: {extract_to}")

            # Delete the zip file after extraction
            os.remove(output_file)
            print(f"Deleted the zip file: {output_file}")
        else:
            print("The downloaded file is not a zip file. Skipping extraction.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except zipfile.BadZipFile:
        print("The file appears to be corrupted or is not a valid zip file.")
    except OSError as e:
        print(f"Error deleting file: {e}")

# Example usage
github_url = "https://github.com/statbiophys/ABGen/releases/download/v1.0.0/"
file_names = ["data.zip","embeddings.zip","gen_seqs.zip"]
output_path = ["./lib/Covid/","./lib/Covid/","./lib/dataset/"]  # Adjust this to where you want to save the file
for i in range(len(file_names)):
    url = github_url + file_names[i]
    output = output_path[i] + file_names[i]
    download_and_unzip_from_github(url, output, extract_to=output_path[i])
