#!/usr/bin/env python3
"""
Script to automatically download datasets from EMBL's BioImage Archive
and extract metadata information.
"""

import os
import re
import requests
import yaml
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from pathlib import Path


class BioImageArchiveDownloader:
    def __init__(self, base_data_folder="data"):
        self.base_data_folder = Path(base_data_folder)
        self.base_data_folder.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_next_dataset_number(self):
        """Get the next available dataset number."""
        existing_folders = [f for f in self.base_data_folder.iterdir() 
                          if f.is_dir() and f.name.startswith('dataset_')]
        if not existing_folders:
            return "001"
        
        numbers = []
        for folder in existing_folders:
            match = re.match(r'dataset_(\d+)', folder.name)
            if match:
                numbers.append(int(match.group(1)))
        
        next_num = max(numbers) + 1 if numbers else 1
        return f"{next_num:03d}"
    
    def extract_accession_from_url(self, url):
        """Extract dataset accession from URL."""
        match = re.search(r'/galleries/(S-[A-Z]+[0-9]+)', url)
        return match.group(1) if match else None
    
    def parse_dataset_page(self, url):
        """Parse the dataset page and extract metadata."""
        print(f"Fetching dataset page: {url}")
        response = self.session.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract basic study information
        metadata = {
            'source_url': url,
            'download_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # Extract study title
        title_elem = soup.find('h1')
        if title_elem:
            metadata['study_title'] = title_elem.get_text(strip=True)
        
        # Extract study information section
        study_info = {}
        
        # Look for organism
        organism_elem = soup.find(string=re.compile(r'Organism', re.I))
        if organism_elem:
            organism_value = organism_elem.find_next('div')
            if organism_value:
                study_info['organism'] = organism_value.get_text(strip=True)
        
        # Look for imaging type
        imaging_elem = soup.find(string=re.compile(r'Imaging type', re.I))
        if imaging_elem:
            imaging_value = imaging_elem.find_next('div')
            if imaging_value:
                study_info['imaging_type'] = imaging_value.get_text(strip=True)
        
        # Look for license
        license_elem = soup.find(string=re.compile(r'License', re.I))
        if license_elem:
            license_value = license_elem.find_next('div')
            if license_value:
                study_info['license'] = license_value.get_text(strip=True)
        
        # Look for author
        author_elem = soup.find(string=re.compile(r'By|Author', re.I))
        if author_elem:
            author_value = author_elem.find_next('div')
            if author_value:
                study_info['author'] = author_value.get_text(strip=True)
        
        # Look for release date
        release_elem = soup.find(string=re.compile(r'Released', re.I))
        if release_elem:
            release_value = release_elem.find_next('div')
            if release_value:
                study_info['release_date'] = release_value.get_text(strip=True)
        
        metadata['study_info'] = study_info
        
        # Extract content information
        content_info = {}
        content_elem = soup.find(string=re.compile(r'Content', re.I))
        if content_elem:
            content_text = content_elem.get_text(strip=True)
            # Extract number of images
            images_match = re.search(r'(\d+)\s+images?', content_text)
            if images_match:
                content_info['total_images'] = int(images_match.group(1))
            
            # Extract number of other files
            files_match = re.search(r'(\d+)\s+other\s+files?', content_text)
            if files_match:
                content_info['other_files'] = int(files_match.group(1))
        
        metadata['content_info'] = content_info
        
        # Extract image information from both tables
        images = []
        tables = soup.find_all('table')
        
        # First, parse the "Viewable images" table (has preview images)
        if len(tables) > 0:
            viewable_table = tables[0]
            print("Parsing viewable images table (with previews)...")
            rows = viewable_table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 6:  # Image ID, Preview, Filename, Dimensions, Download Size, Actions
                    image_info = self._parse_image_row(cells, url, has_preview=True)
                    if image_info:
                        images.append(image_info)
        
        # Then, try to parse the "All images" table (complete list)
        if len(tables) > 1:
            all_images_table = tables[1]
            print("Parsing all images table (complete list)...")
            rows = all_images_table.find_all('tr')[1:]  # Skip header row
            print(f"Found {len(rows)} rows in all images table")
            
            if len(rows) == 0:
                print("All images table appears to be empty (likely loaded dynamically)")
                print("Only images with previews are available for download")
            else:
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 4:  # Image ID, Filename, Download Size, Actions
                        image_id_text = cells[0].get_text(strip=True)
                        
                        # Check if we already have this image from the viewable table
                        existing_image = next((img for img in images if img.get('image_id') == image_id_text), None)
                        
                        if not existing_image:
                            # Parse this row and add it
                            image_info = self._parse_image_row(cells, url, has_preview=False)
                            if image_info:
                                images.append(image_info)
                        else:
                            # We already have this image with preview, skip
                            print(f"Image {image_id_text} already exists with preview, skipping from all images table")
        
        # Sort images by image_id for consistent ordering
        images.sort(key=lambda x: int(x.get('image_id', '0').replace('IM', '')) if x.get('image_id', '').replace('IM', '').isdigit() else 999)
        
        metadata['images'] = images
        print(f"Found {len(images)} images in the dataset")
        
        return metadata
    
    def _parse_image_row(self, cells, url, has_preview=False):
        """Parse a table row to extract image information."""
        image_info = {}
        
        # Image ID (first column)
        image_id_text = cells[0].get_text(strip=True)
        if image_id_text:
            image_info['image_id'] = image_id_text
        
        # Filename (second column in all images table, third in viewable table)
        filename_col = 2 if has_preview else 1
        if len(cells) > filename_col:
            filename_text = cells[filename_col].get_text(strip=True)
            if filename_text:
                image_info['filename'] = filename_text
        
        # Dimensions (third column in viewable table, not available in all images table)
        if has_preview and len(cells) > 3:
            dimensions_text = cells[3].get_text(strip=True)
            if dimensions_text and dimensions_text != 'Unavailable':
                # Parse dimensions like (1, 4, 3, 2160, 2160)
                dims_match = re.search(r'\(([^)]+)\)', dimensions_text)
                if dims_match:
                    dims = [int(x.strip()) for x in dims_match.group(1).split(',')]
                    image_info['dimensions'] = {
                        'T': dims[0] if len(dims) > 0 else 1,
                        'C': dims[1] if len(dims) > 1 else 1,
                        'Z': dims[2] if len(dims) > 2 else 1,
                        'Y': dims[3] if len(dims) > 3 else 1,
                        'X': dims[4] if len(dims) > 4 else 1
                    }
        
        # Download URL (from actions column)
        actions_col = 5 if has_preview else 3
        if len(cells) > actions_col:
            actions_cell = cells[actions_col]
            download_links = actions_cell.find_all('a', href=True)
            for link in download_links:
                href = link.get('href')
                if href and ('download' in href.lower() or 'files' in href.lower()):
                    image_info['download_url'] = href
                    break
        
        # Preview image URL (only in viewable table)
        if has_preview and len(cells) > 1:
            preview_cell = cells[1]  # Preview column
            preview_img = preview_cell.find('img')
            if preview_img:
                preview_src = preview_img.get('src')
                if preview_src:
                    # Convert relative URL to absolute
                    if preview_src.startswith('/'):
                        image_info['preview_url'] = 'https://www.ebi.ac.uk' + preview_src
                    else:
                        image_info['preview_url'] = urljoin(url, preview_src)
        
        return image_info if image_info else None
    
    def download_image(self, image_url, local_path):
        """Download a single image."""
        print(f"Downloading: {image_url}")
        response = self.session.get(image_url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {local_path}")
        return local_path
    
    def download_dataset(self, dataset_url, image_id=None, download_files=True):
        """Download a complete dataset from BioImage Archive."""
        # Get next dataset number
        dataset_num = self.get_next_dataset_number()
        dataset_folder = self.base_data_folder / f"dataset_{dataset_num}"
        dataset_folder.mkdir(exist_ok=True)
        
        print(f"Creating dataset folder: {dataset_folder}")
        
        # Extract accession
        accession = self.extract_accession_from_url(dataset_url)
        if not accession:
            raise ValueError("Could not extract accession from URL")
        
        # Parse dataset page
        metadata = self.parse_dataset_page(dataset_url)
        metadata['accession'] = accession
        metadata['dataset_number'] = dataset_num
        metadata['download_files'] = download_files
        
        # Download images (if requested)
        downloaded_files = []
        images_to_download = metadata['images']
        
        # Filter by image_id if provided
        if image_id:
            images_to_download = [img for img in images_to_download if img.get('image_id') == image_id]
            if not images_to_download:
                available_ids = [img.get('image_id') for img in metadata['images']]
                print(f"Available Image IDs: {available_ids}")
                print("Note: Only images with preview thumbnails are available for download.")
                print("The full list of 1170 images is loaded dynamically and not accessible via simple HTTP requests.")
                raise ValueError(f"Image ID '{image_id}' not found. Available IDs: {available_ids}")
            print(f"Filtering to Image ID: {image_id}")
        else:
            # If no image_id specified, download first image
            images_to_download = images_to_download[:1]
            print("No Image ID specified, downloading first image")
        
        # Download preview images as part of metadata (only if they exist)
        preview_files = []
        for i, image_info in enumerate(images_to_download):
            if 'preview_url' in image_info and image_info['preview_url']:
                preview_filename = f"preview_{image_info['image_id']}.jpg"
                preview_path = dataset_folder / preview_filename
                
                try:
                    print(f"Downloading preview: {image_info['image_id']}")
                    self.download_image(image_info['preview_url'], preview_path)
                    preview_files.append({
                        'filename': preview_filename,
                        'local_path': str(preview_path),
                        'image_id': image_info['image_id'],
                        'preview_url': image_info['preview_url']
                    })
                except Exception as e:
                    print(f"Failed to download preview for {image_info['image_id']}: {e}")
            else:
                print(f"No preview available for {image_info.get('image_id', 'Unknown')}")
        
        if download_files:
            print("Downloading image files...")
            for i, image_info in enumerate(images_to_download):
                if 'download_url' in image_info:
                    filename = image_info['filename']
                    local_path = dataset_folder / filename
                    
                    # Create directory structure if needed
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        self.download_image(image_info['download_url'], local_path)
                        downloaded_files.append({
                            'filename': filename,
                            'local_path': str(local_path),
                            'image_info': image_info
                        })
                    except Exception as e:
                        print(f"Failed to download {filename}: {e}")
        else:
            print("Skipping file downloads - metadata only mode")
            # Still track which files would be downloaded
            for i, image_info in enumerate(images_to_download):
                if 'download_url' in image_info:
                    downloaded_files.append({
                        'filename': image_info['filename'],
                        'local_path': 'not_downloaded',
                        'image_info': image_info
                    })
        
        metadata['downloaded_files'] = downloaded_files
        metadata['preview_files'] = preview_files
        
        # Save metadata
        metadata_file = dataset_folder / f"dataset_{dataset_num}.yaml"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Metadata saved to: {metadata_file}")
        print(f"Dataset {dataset_num} completed successfully!")
        
        return dataset_folder, metadata_file
    
    def list_available_images(self, dataset_url):
        """List all available image IDs for a dataset."""
        print(f"Fetching available images from: {dataset_url}")
        metadata = self.parse_dataset_page(dataset_url)
        
        print(f"\nAvailable Image IDs for {metadata.get('accession', 'Unknown')}:")
        print("-" * 50)
        
        for i, image in enumerate(metadata['images'], 1):
            image_id = image.get('image_id', 'Unknown')
            filename = image.get('filename', 'Unknown')
            dimensions = image.get('dimensions', {})
            
            print(f"{i:2d}. {image_id:8s} - {filename}")
            if dimensions:
                dims_str = f"({dimensions.get('T', 1)}, {dimensions.get('C', 1)}, {dimensions.get('Z', 1)}, {dimensions.get('Y', 1)}, {dimensions.get('X', 1)})"
                print(f"     Dimensions: {dims_str}")
        
        return [img.get('image_id') for img in metadata['images']]
    
    def anonymize_dataset(self, dataset_folder, metadata_file):
        """Anonymize the dataset by renaming folders and files to generic names within the same folder."""
        print(f"Anonymizing dataset: {dataset_folder}")
        
        # Get the dataset number from the folder name
        dataset_num = dataset_folder.name.split('_')[1]  # Extract number from "dataset_001"
        
        # Track renamed files
        anonymized_files = []
        
        # Process downloaded files - rename them within the same dataset folder
        for item in dataset_folder.rglob('*'):
            if item.is_file() and item.name != f"dataset_{dataset_num}.yaml":
                # Get relative path from original dataset folder
                rel_path = item.relative_to(dataset_folder)
                
                # Create anonymized filename
                if rel_path.parent == Path('.'):
                    # File is in root of dataset folder
                    anonymized_filename = f"dataset_{dataset_num}{item.suffix}"
                else:
                    # File is in subfolder - flatten to root with dataset number
                    anonymized_filename = f"dataset_{dataset_num}{item.suffix}"
                
                # Create new path within the same dataset folder
                anonymized_path = dataset_folder / anonymized_filename
                
                # Move (rename) the file within the same folder
                item.rename(anonymized_path)
                
                anonymized_files.append({
                    'original_path': str(rel_path),
                    'anonymized_path': str(anonymized_path.relative_to(self.base_data_folder)),
                    'anonymized_filename': anonymized_filename
                })
                
                print(f"Renamed: {rel_path} -> {anonymized_filename}")
        
        # Remove empty subfolders after renaming files
        for item in dataset_folder.rglob('*'):
            if item.is_dir() and item != dataset_folder:
                # Check if directory is empty
                try:
                    if not any(item.iterdir()):  # Directory is empty
                        item.rmdir()
                        print(f"Removed empty folder: {item.relative_to(dataset_folder)}")
                except OSError:
                    # Directory not empty or other error, skip
                    pass
        
        # Update metadata with anonymization info
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        
        metadata['anonymized'] = True
        metadata['anonymized_files'] = anonymized_files
        metadata['original_dataset_folder'] = str(dataset_folder.relative_to(self.base_data_folder))
        metadata['anonymized_dataset_folder'] = str(dataset_folder.relative_to(self.base_data_folder))
        
        # Save updated metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Anonymized dataset: {dataset_folder}")
        print(f"Anonymized metadata: {metadata_file}")
        
        return dataset_folder, metadata_file


def main():
    """Main function to test the downloader."""
    downloader = BioImageArchiveDownloader()
    
    # Test with S-BIAD7 dataset
    test_url = "https://www.ebi.ac.uk/bioimage-archive/galleries/S-BIAD7.html"
    
    try:
        # Example: Download specific image by ID
        dataset_folder, metadata_file = downloader.download_dataset(
            test_url, 
            image_id="IM76",  # Specify which image to download
            download_files=False  
        )
        print(f"\nSuccess! Dataset processed: {dataset_folder}")
        print(f"Metadata saved to: {metadata_file}")
        
        # Anonymize the dataset
        print("\n" + "="*50)
        print("ANONYMIZING DATASET")
        print("="*50)
        anonymized_folder, anonymized_metadata = downloader.anonymize_dataset(dataset_folder, metadata_file)
        print(f"\nAnonymized dataset: {anonymized_folder}")
        print(f"Anonymized metadata: {anonymized_metadata}")
        
    except Exception as e:
        print(f"Error processing dataset: {e}")


if __name__ == "__main__":
    main()
