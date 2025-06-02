import requests
import os
import zipfile
import shutil
from pathlib import Path

def setup_covid_dataset(record_id, base_dir="./lib"):
    """
    Enhanced version with verification and better error handling
    """
    
    base_path = Path(base_dir)
    covid_dir = base_path / "Covid"
    dataset_dir = base_path / "dataset"
    temp_downloads = base_path / "temp_downloads"
    
    # Create directories
    for dir_path in [covid_dir, dataset_dir, temp_downloads]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # File mapping with expected contents
    file_config = {
        "data.zip": {
            "destination": covid_dir,
            "description": "COVID data files"
        },
        "embeddings.zip": {
            "destination": covid_dir,
            "description": "Embedding vectors"
        },
        "alphafold_pred.zip": {
            "destination": dataset_dir,
            "description": "AlphaFold predictions"
        },
        "gen_seqs.zip": {
            "destination": dataset_dir,
            "description": "Generated sequences"
        }
    }
    
    print("=== COVID Dataset Setup ===")
    print(f"Base directory: {base_path.absolute()}")
    print(f"Target structure:")
    print(f"  .lib/covid/ ← data.zip, embeddings.zip")
    print(f"  .lib/dataset/ ← alphafold_pred.zip, gen_seqs.zip")
    
    try:
        # Download files
        print(f"\n=== DOWNLOADING FROM ZENODO {record_id} ===")
        downloaded_files = download_and_verify(record_id, temp_downloads, file_config)
        
        # Extract and organize
        print(f"\n=== EXTRACTING AND ORGANIZING ===")
        extract_and_organize(downloaded_files, file_config)
        
        # Verify final structure
        print(f"\n=== VERIFICATION ===")
        verify_final_structure(covid_dir, dataset_dir)
        
        # Cleanup
        shutil.rmtree(temp_downloads)
        print(f"\n✓ Setup complete! Data ready for analysis.")
        
        return {
            'covid_dir': covid_dir,
            'dataset_dir': dataset_dir,
            'base_path': base_path
        }
        
    except Exception as e:
        print(f"\n✗ Error during setup: {e}")
        return None

def download_and_verify(record_id, download_dir, file_config):
    """Download and verify all required files are present"""
    
    # Get Zenodo metadata
    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url)
    response.raise_for_status()
    
    record_data = response.json()
    available_files = {f['key']: f for f in record_data['files']}
    
    print(f"Zenodo record: {record_data.get('title', 'Unknown')}")
    print(f"Available files: {list(available_files.keys())}")
    
    # Check if all required files are available
    missing_files = set(file_config.keys()) - set(available_files.keys())
    if missing_files:
        raise Exception(f"Missing required files: {missing_files}")
    
    downloaded_files = {}
    
    # Download each required file
    for filename in file_config.keys():
        file_info = available_files[filename]
        file_url = file_info['links']['self']
        file_size = file_info['size']
        file_path = download_dir / filename
        
        print(f"\nDownloading {filename}...")
        print(f"  Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"  Purpose: {file_config[filename]['description']}")
        
        # Download with progress
        response = requests.get(file_url, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                
                # Progress bar
                if downloaded % (512 * 1024) == 0:  # Every 512KB
                    percent = (downloaded / file_size) * 100
                    bar_length = 30
                    filled = int(bar_length * downloaded // file_size)
                    bar = '█' * filled + '-' * (bar_length - filled)
                    print(f"  [{bar}] {percent:.1f}%", end='\r')
        
        print(f"{filename} downloaded successfully")
        downloaded_files[filename] = file_path
    
    return downloaded_files

def extract_and_organize(downloaded_files, file_config):
    """Extract files to their designated locations"""
    
    for filename, file_path in downloaded_files.items():
        config = file_config[filename]
        destination = config['destination']
        
        print(f"\nExtracting {filename}...")
        print(f"  Destination: {destination}")
        
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Get file list
                files_in_zip = zip_ref.namelist()
                
                # Check for potential overwrites
                existing_files = []
                for zip_file in files_in_zip:
                    target_path = destination / zip_file
                    if target_path.exists():
                        existing_files.append(zip_file)
                
                if existing_files:
                    print(f"  Warning: Will overwrite {len(existing_files)} existing files")
                
                # Extract all files
                zip_ref.extractall(destination)
                
                print(f"Extracted {len(files_in_zip)} files")
                
                # Show sample of extracted files
                sample_files = files_in_zip[:3]
                print(f"  Sample files: {', '.join(sample_files)}")
                if len(files_in_zip) > 3:
                    print(f"  (and {len(files_in_zip) - 3} more files)")
                
        except zipfile.BadZipFile:
            print(f" {filename} is not a valid ZIP file")
        except Exception as e:
            print(f" Error extracting {filename}: {e}")

def verify_final_structure(covid_dir, dataset_dir):
    """Verify the final directory structure"""
    
    def count_files_recursive(directory):
        return sum(1 for _ in directory.rglob('*') if _.is_file())
    
    covid_files = count_files_recursive(covid_dir)
    dataset_files = count_files_recursive(dataset_dir)
    
    print(f"Final structure verification:")
    print(f"  {covid_dir}: {covid_files} files")
    print(f"  {dataset_dir}: {dataset_files} files")
    
    # List top-level contents
    print(f"\nCOVID directory contents:")
    for item in sorted(covid_dir.iterdir()):
        if item.is_file():
            print(f" {item.name}")
        else:
            file_count = count_files_recursive(item)
            print(f" {item.name}/ ({file_count} files)")
    
    print(f"\nDataset directory contents:")
    for item in sorted(dataset_dir.iterdir()):
        if item.is_file():
            print(f" {item.name}")
        else:
            file_count = count_files_recursive(item)
            print(f" {item.name}/ ({file_count} files)")


if __name__ == '__main__':
    # Usage
    setup_covid_dataset("15571259")