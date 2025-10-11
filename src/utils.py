"""
utils.py - Optimized utility functions for downloading and processing product images
"""

import pandas as pd
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import time
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_single_image(row, output_path, headers, max_retries=3, delay=0.3):
    """
    Download a single image.
    
    Parameters:
    -----------
    row : pandas.Series
        Row containing sample_id and image_link
    output_path : Path
        Directory to save images
    headers : dict
        Request headers
    max_retries : int
        Maximum retry attempts
    delay : float
        Delay between retries
    
    Returns:
    --------
    tuple
        (success: bool, sample_id: int, error_msg: str or None)
    """
    sample_id = row['sample_id']
    image_url = row['image_link']
    
    # Check if image already exists
    for ext in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
        if (output_path / f"{sample_id}.{ext}").exists():
            return (True, sample_id, "already_exists")
    
    # Skip if URL is empty or NaN
    if pd.isna(image_url) or image_url == '' or not isinstance(image_url, str):
        return (False, sample_id, "invalid_url")
    
    # Try downloading with retries
    for attempt in range(max_retries):
        try:
            response = requests.get(
                image_url, 
                headers=headers, 
                timeout=10,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Open and verify image
            img = Image.open(BytesIO(response.content))
            img.verify()
            
            # Re-open after verify
            img = Image.open(BytesIO(response.content))
            
            # Convert RGBA to RGB if needed
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Determine file extension
            img_format = img.format.lower() if img.format else 'jpg'
            if img_format == 'jpeg':
                img_format = 'jpg'
            
            filename = f"{sample_id}.{img_format}"
            filepath = output_path / filename
            
            # Save image
            img.save(filepath, quality=90)
            
            return (True, sample_id, None)
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                return (False, sample_id, type(e).__name__)
    
    return (False, sample_id, "max_retries_exceeded")


def download_images(csv_file, output_dir='data/images', max_workers=10, max_retries=3, 
                    delay=0.3, batch_size=None):
    """
    Download images using parallel processing for much faster downloads.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing 'sample_id' and 'image_link' columns
    output_dir : str
        Directory where images will be saved
    max_workers : int
        Number of parallel download threads (default: 10)
        Increase to 20-30 for even faster downloads if your network can handle it
    max_retries : int
        Maximum number of retry attempts per image (default: 3)
    delay : float
        Delay in seconds between retries (default: 0.3)
    batch_size : int
        Process only a subset of images (useful for testing)
    
    Returns:
    --------
    dict
        Dictionary with download statistics
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Limit batch size if specified
    if batch_size is not None:
        df = df.head(batch_size)
        logger.info(f"Processing first {batch_size} images only")
    
    # Initialize statistics
    stats = {
        'total': len(df),
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'failed_ids': []
    }
    
    # Request headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
    }
    
    # Create partial function with fixed parameters
    download_func = partial(
        download_single_image,
        output_path=output_path,
        headers=headers,
        max_retries=max_retries,
        delay=delay
    )
    
    # Download images in parallel
    logger.info(f"Starting download with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(download_func, row): row['sample_id'] 
                  for _, row in df.iterrows()}
        
        # Process completed downloads with progress bar
        with tqdm(total=len(futures), desc="Downloading images") as pbar:
            for future in as_completed(futures):
                success, sample_id, error_msg = future.result()
                
                if success:
                    stats['successful'] += 1
                    if error_msg == "already_exists":
                        stats['skipped'] += 1
                else:
                    stats['failed'] += 1
                    stats['failed_ids'].append(sample_id)
                
                pbar.update(1)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Download Summary for {csv_file}:")
    logger.info(f"Total images: {stats['total']}")
    logger.info(f"Successfully downloaded: {stats['successful'] - stats['skipped']}")
    logger.info(f"Already existed (skipped): {stats['skipped']}")
    logger.info(f"Failed downloads: {stats['failed']}")
    logger.info(f"Success rate: {stats['successful']/stats['total']*100:.2f}%")
    logger.info(f"{'='*60}\n")
    
    # Save failed IDs
    if stats['failed_ids']:
        failed_file = output_path / 'failed_downloads.txt'
        with open(failed_file, 'w') as f:
            f.write('\n'.join(map(str, stats['failed_ids'])))
        logger.info(f"Failed sample IDs saved to {failed_file}")
    
    return stats


def retry_failed_downloads(csv_file, output_dir='data/images', max_workers=10, 
                           max_retries=5, delay=0.5):
    """
    Retry downloading failed images with parallel processing.
    
    Parameters:
    -----------
    csv_file : str
        Path to the original CSV file
    output_dir : str
        Directory where images are/will be saved
    max_workers : int
        Number of parallel workers
    max_retries : int
        Maximum retry attempts
    delay : float
        Delay between retries
    
    Returns:
    --------
    dict
        Dictionary with retry statistics
    """
    
    output_path = Path(output_dir)
    failed_ids_file = output_path / 'failed_downloads.txt'
    
    if not failed_ids_file.exists():
        logger.info(f"No failed downloads file found at {failed_ids_file}")
        return None
    
    # Read failed IDs
    with open(failed_ids_file, 'r') as f:
        failed_ids = [int(line.strip()) for line in f.readlines() if line.strip()]
    
    if not failed_ids:
        logger.info("No failed IDs to retry")
        return None
    
    logger.info(f"Retrying {len(failed_ids)} failed downloads...")
    
    # Read CSV and filter for failed IDs
    df = pd.read_csv(csv_file)
    df_failed = df[df['sample_id'].isin(failed_ids)]
    
    # Create temporary CSV
    temp_csv = output_path / 'temp_retry.csv'
    df_failed.to_csv(temp_csv, index=False)
    
    # Download with more workers and retries
    stats = download_images(
        str(temp_csv), 
        output_dir, 
        max_workers=max_workers,
        max_retries=max_retries, 
        delay=delay
    )
    
    # Clean up
    temp_csv.unlink()
    
    return stats


def verify_downloads(csv_file, output_dir='data/images', check_corruption=True):
    """
    Verify that all images have been downloaded successfully.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    output_dir : str
        Directory where images are saved
    check_corruption : bool
        Whether to check for corrupted images (slower but thorough)
    
    Returns:
    --------
    dict
        Dictionary with verification results
    """
    
    df = pd.read_csv(csv_file)
    output_path = Path(output_dir)
    
    missing_ids = []
    corrupted_ids = []
    successful = 0
    
    logger.info(f"Verifying downloads for {csv_file}...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying images"):
        sample_id = row['sample_id']
        
        # Check for common image extensions
        found = False
        for ext in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
            filepath = output_path / f"{sample_id}.{ext}"
            if filepath.exists():
                found = True
                
                # Optionally check for corruption
                if check_corruption:
                    try:
                        with Image.open(filepath) as img:
                            img.verify()
                        successful += 1
                    except Exception:
                        corrupted_ids.append(sample_id)
                else:
                    successful += 1
                break
        
        if not found:
            missing_ids.append(sample_id)
    
    results = {
        'total': len(df),
        'successful': successful,
        'missing': len(missing_ids),
        'corrupted': len(corrupted_ids),
        'missing_ids': missing_ids,
        'corrupted_ids': corrupted_ids
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Verification Results for {csv_file}:")
    logger.info(f"Total samples: {results['total']}")
    logger.info(f"Successfully verified: {results['successful']}")
    logger.info(f"Missing images: {results['missing']}")
    logger.info(f"Corrupted images: {results['corrupted']}")
    logger.info(f"Verification rate: {results['successful']/results['total']*100:.2f}%")
    logger.info(f"{'='*60}\n")
    
    return results


def get_image_path(sample_id, images_dir='data/images'):
    """Get the full path to an image given its sample_id."""
    output_path = Path(images_dir)
    
    for ext in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
        filepath = output_path / f"{sample_id}.{ext}"
        if filepath.exists():
            return filepath
    
    return None


def load_image(sample_id, images_dir='data/images', target_size=None):
    """Load an image given its sample_id."""
    filepath = get_image_path(sample_id, images_dir)
    
    if filepath is None:
        return None
    
    try:
        img = Image.open(filepath)
        
        if target_size is not None:
            img = img.resize(target_size, Image.LANCZOS)
        
        return img
    
    except Exception as e:
        logger.error(f"Error loading image {sample_id}: {str(e)}")
        return None
