import cv2
import os
from frames_processing import process_multitask
import os
from dotenv import load_dotenv
from sensing_garden_client import SensingGardenClient
import json
import requests
import tempfile
from tqdm import tqdm
import argparse
import time
import signal
import sys

def extract_frames(input_path, output_dir="./frames", interval_seconds=None):
    """
    Extract frames from a video.
    
    Args:
        input_path: Path to input video
        output_dir: Output directory for frames
        interval_seconds: If provided, extract frames at this interval (in seconds).
                          If None, extract all frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    vidcap = cv2.VideoCapture(input_path)
    
    if not vidcap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"Extracting frames from {input_path}")
    print(f"FPS: {fps}, Total frames: {frame_count}, Duration: {duration:.2f} seconds")
    
    count = 0
    
    if interval_seconds is not None:
        print(f"Extracting frames at {interval_seconds} second intervals")
        current_ms = 0
        
        while True:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, current_ms)
            success, image = vidcap.read()
            if not success:
                break
            output_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(output_path, image)
            count += 1
            current_ms = count * 1000 * interval_seconds
            if count % 10 == 0:
                print(f"Extracted {count} frames...")

    else:
        print("Extracting all frames")
        while True:
            success, image = vidcap.read()
            if not success:
                break
            output_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(output_path, image)
            count += 1
            
            if count % 100 == 0:
                print(f"Processed {count} frames...")
    
    vidcap.release()
    print(f"Extraction complete. Saved {count} frames to {output_dir}")

def json_to_database(sgc, device_id, model_id, json_file, frame_dir, timestamp):
    """
    Upload detection and classification results from JSON file to the Sensing Garden database.
    
    Args:
        json_file: Path to the JSON file containing detection/classification results
    """
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} frames from {json_file}")
    
    frame_names = sorted(data.keys(), key=lambda x: int(x.split('_')[1].split('.')[0]))
    for i, frame_name in enumerate(tqdm(frame_names, desc="Uploading classifications")):
        detections = data[frame_name]
        if not isinstance(detections, list):
            continue  # Skip if not a list of detections
        
        frame_path = os.path.join(frame_dir, frame_name)
        
        if not os.path.exists(frame_path):
            print(f"Warning: Image file {frame_path} not found")
            continue

        # Create a unique timestamp for each frame by adding frame index
        frame_number = int(frame_name.split('_')[1].split('.')[0])
        frame_timestamp = f"{timestamp}_{frame_number:04d}"

        with open(frame_path, "rb") as f:
            image_data = f.read()

        for detection in detections:
            try:
                
                sgc.classifications.add(
                    device_id=device_id,
                    model_id=model_id,
                    image_data=image_data,
                    family=detection["family"],
                    genus=detection["genus"],
                    species=detection["species"],
                    family_confidence=detection["family_confidence"],
                    genus_confidence=detection["genus_confidence"],
                    species_confidence=detection["species_confidence"],
                    timestamp=frame_timestamp,
                    bounding_box=detection["bbox"],
                    track_id=detection["track_id"]
                )
                
                # Removed individual print for each upload
            except Exception as e:
                print(f"Error uploading data for {frame_name}: {str(e)}")
    
    print(f"Completed uploading data from {json_file}")

def download_videos(sgc, device_id, output_dir, limit=10, skip_identifiers=None):
    """
    Download videos from Sensing Garden API for a specific device.
    
    Args:
        device_id: The device ID to fetch videos for
        output_dir: Directory to save temporary video files
        limit: Maximum number of videos to fetch
        skip_identifiers: Set of video identifiers to skip (already processed)
    
    Returns:
        List of dictionaries with video info and temporary file paths
    """

    if skip_identifiers is None:
        skip_identifiers = set()

    video_count = sgc.videos.count(device_id=device_id)

    print(f"Video count: {video_count}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    videos_response = sgc.videos.fetch(
        device_id=device_id,
        limit=limit
    )
    
    print(f"Found {len(videos_response.get('items', []))} videos")
    
    downloaded_videos = []
    skipped_count = 0
    
    for i, video in enumerate(videos_response.get('items', [])):
        try:
            video_id = video.get('id')
            timestamp = video.get('timestamp')
            url = video.get('url')
            
            video_key = video.get('video_key')
            if not video_key and 'metadata' in video and isinstance(video['metadata'], dict):
                video_key = video['metadata'].get('video_key')
            if not video_key and 's3' in video and isinstance(video['s3'], dict):
                video_key = video['s3'].get('key')
            
            if timestamp:
                formatted_ts = timestamp.replace(':', '-').replace(' ', '_').replace('.', '-')
                video_identifier = f"{formatted_ts}"
            else:
                video_identifier = f"video_{i+1}"
                
            if video_identifier in skip_identifiers:
                print(f"Skipping video {i+1}/{len(videos_response.get('items', []))}: {video_identifier} (already processed)")
                skipped_count += 1
                continue
            
            if not url and video_key:
                try:
                    bucket_name = "scl-sensing-garden-videos"  # From the VideosClient class
                    presigned_url = sgc.videos._s3_client.generate_presigned_url(
                        'get_object',
                        Params={
                            'Bucket': bucket_name,
                            'Key': video_key
                        },
                        ExpiresIn=3600  
                    )
                    url = presigned_url
                except Exception as e:
                    print(f"Error generating presigned URL for video {i+1}: {str(e)}")
                    continue
            
            if not url:
                print(f"No URL available for video {i+1}, skipping")
                continue
                
            print(f"Downloading video {i+1}/{len(videos_response.get('items', []))}: {video_identifier}")
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_video_path = temp_file.name
                
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
            
            downloaded_videos.append({
                'id': video_id,
                'timestamp': timestamp,
                'identifier': video_identifier,
                'temp_path': temp_video_path
            })
            
            print(f"Successfully downloaded video to: {temp_video_path}")
            
        except Exception as e:
            print(f"Error downloading video {i+1}: {str(e)}")
    
    print(f"Downloaded {len(downloaded_videos)} videos, skipped {skipped_count} already processed videos")
    return downloaded_videos

def main_processing_loop(args, frames_base_dir, species_names, existing_dirs, download_limit=10):
    """
    Main processing function that downloads videos, extracts frames, and processes them.
    This function will be called repeatedly on a schedule.
    """
    OUTPUT_PATH = args.output_path
    YOLO_WEIGHTS = args.yolo_weights
    HIERARCHICAL_WEIGHTS = args.hierarchical_weights
    DEVICE_ID = args.device_id
    MODEL_ID = args.model_id
    time_interval = args.time_interval
    
    # Load environment variables and initialize client
    sgc = SensingGardenClient(
        base_url = os.environ.get('API_BASE_URL'),
        api_key = os.environ.get('SENSING_GARDEN_API_KEY'),
        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY'),
        aws_region = os.environ.get('AWS_REGION', 'us-east-1')
    )
    
    # Get list of already processed directories to avoid reprocessing
    store_file = os.path.join(OUTPUT_PATH, "store.txt")
    processed_dirs = set()
    if os.path.exists(store_file):
        with open(store_file, 'r') as f:
            processed_dirs = {line.strip() for line in f.readlines()}
    
    # Combine existing directories with processed directories to avoid redownloading
    skip_identifiers = existing_dirs.union(processed_dirs)
    
    # Download new videos with the current limit
    print(f"Downloading videos with limit: {download_limit}")
    downloaded_videos = download_videos(sgc, DEVICE_ID, OUTPUT_PATH, limit=download_limit, skip_identifiers=skip_identifiers)
    
    # Process each downloaded video
    for video in downloaded_videos:
        video_frames_dir = os.path.join(frames_base_dir, video['identifier'])
        os.makedirs(video_frames_dir, exist_ok=True)
        
        print(f"Extracting frames from video {video['identifier']} to {video_frames_dir}")
        extract_frames(
            input_path=video['temp_path'],
            output_dir=video_frames_dir,
            interval_seconds=time_interval
        )
        
        os.unlink(video['temp_path'])
        print(f"Removed temporary video file: {video['temp_path']}")
    
    if downloaded_videos:
        print(f"Processing complete. Frames saved to {frames_base_dir}")
        # Update the existing directories set with the newly processed videos
        for video in downloaded_videos:
            existing_dirs.add(video['identifier'])
    else:
        print("No new videos to process")
    
    # Process all unprocessed frame directories
    for dir_name in os.listdir(frames_base_dir):
        dir_path = os.path.join(frames_base_dir, dir_name)
        if os.path.isdir(dir_path) and dir_name not in processed_dirs:
            print(f"Processing frames in {dir_name}")
            _, output_file = process_multitask(species_names, dir_path, YOLO_WEIGHTS, HIERARCHICAL_WEIGHTS, OUTPUT_PATH)
            
            if output_file:
                with open(store_file, 'a') as f:
                    f.write(f"{dir_name}\n")
                print(f"Added {dir_name} to processed directories")
                processed_dirs.add(dir_name)
                
                timestamp = dir_name
                
                # Check if any classifications exist for this video
                timestamp_prefix = timestamp
                existing_classifications = sgc.classifications.fetch(
                    device_id=DEVICE_ID,
                    model_id=MODEL_ID,
                    limit=1,
                    start_time=timestamp_prefix,
                    end_time=timestamp_prefix + "Z"
                )
                
                if existing_classifications and len(existing_classifications.get('items', [])) > 0:
                    print(f"Skipping upload for {dir_name}: classifications already exist with this timestamp prefix")
                else:
                    print(f"Uploading classifications for {dir_name}")
                    json_to_database(sgc, DEVICE_ID, MODEL_ID, output_file, dir_path, timestamp)

def signal_handler(sig, frame):
    """Handle graceful shutdown when receiving SIGINT (Ctrl+C)"""
    print("\nShutting down gracefully. Waiting for current processing to complete...")
    sys.exit(0)

def test_processing(video_path, output_path, species_txt, yolo_weights, hierarchical_weights):
    """
    Test processing function that extracts frames from a video, runs object detection, and saves the results.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to the output directory for processed data
        species_txt: Path to the text file containing species names
        yolo_weights: Path to the YOLO weights file
        hierarchical_weights: Path to the hierarchical classifier weights file
    """
    # Create output directory for extracted frames
    frames_output_dir = os.path.join(output_path, "extracted_frames")
    os.makedirs(frames_output_dir, exist_ok=True)
    
    print(f"Created frames directory at: {frames_output_dir}")
    # Extract frames from the video
    extract_frames(video_path, frames_output_dir)

    # Load species names from file
    species_names = []
    try:
        with open(species_txt, 'r') as f:
            species_names = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(species_names)} species names from {species_txt}")
    except Exception as e:
        print(f"Error loading species file: {str(e)}. Using empty species list.")

    # Process the frames with the object detection model
    _, output_file = process_multitask(species_names, frames_output_dir, yolo_weights, hierarchical_weights, output_path)
    
    print(f"Processing complete. Results saved to: {output_file}")
    

if __name__ == "__main__":
    test_processing(video_path="/mnt/f/mit/sensing-garden-processing/2025-05-06T12-01-30-982203.mp4",
                    output_path="/mnt/f/mit/sensing-garden-processing/output",
                    species_txt="/mnt/f/mit/sensing-garden-processing/london_invertebrates.txt",
                    yolo_weights="/mnt/f/mit/sensing-garden-processing/small-generic.pt",
                    hierarchical_weights="/mnt/f/mit/sensing-garden-processing/london_141.pt")
    # # Set up command line argument parsing
    # parser = argparse.ArgumentParser(description='Process videos from Sensing Garden API and run object detection.')
    # parser.add_argument('--output-path', default="./output",
    #                     help='Path to output directory for processed data')
    # parser.add_argument('--yolo-weights', default="./small-generic.pt",
    #                     help='Path to YOLO weights file')
    # parser.add_argument('--hierarchical-weights', default="./best_bjerge.pt",
    #                     help='Path to hierarchical classifier weights file')
    # parser.add_argument('--device-id', default="b8f2ed92a70e5df3",
    #                     help='Device ID for the Sensing Garden API')
    # parser.add_argument('--model-id', default="best_bjerge.pt",
    #                     help='Model ID for the Sensing Garden API')
    # parser.add_argument('--time-interval', type=float, default=None,
    #                     help='Time interval in seconds between extracted frames (None for all frames)')
    # parser.add_argument('--species-file', default=None,
    #                     help='Path to a text file with each species name on a separate line')
    # parser.add_argument('--interval-hours', type=float, default=1.0,
    #                     help='Run interval in hours (default: 1)')
    
    # args = parser.parse_args()
    
    # OUTPUT_PATH = args.output_path
    
    # os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # # Register signal handler for graceful shutdown
    # signal.signal(signal.SIGINT, signal_handler)
    
    # frames_base_dir = os.path.join(OUTPUT_PATH, "extracted_frames")
    # os.makedirs(frames_base_dir, exist_ok=True)
    
    # # Load species names from file if provided, otherwise use default list
    # default_species_names = [
    #     "Coccinella septempunctata", "Apis mellifera", "Bombus lapidarius", "Bombus terrestris",
    #     "Eupeodes corollae", "Episyrphus balteatus", "Aglais urticae", "Vespula vulgaris",
    #     "Eristalis tenax"
    # ]
    
    # species_names = default_species_names
    # if args.species_file:
    #     try:
    #         with open(args.species_file, 'r') as f:
    #             species_names = [line.strip() for line in f if line.strip()]
    #         print(f"Loaded {len(species_names)} species names from {args.species_file}")
    #     except Exception as e:
    #         print(f"Error loading species file: {str(e)}. Using default species list.")
    
    # existing_dirs = set(os.listdir(frames_base_dir)) if os.path.exists(frames_base_dir) else set()
    # # print(f"Found {len(existing_dirs)} already processed video directories")

    # load_dotenv()
    
    # interval_seconds = args.interval_hours * 3600
    
    # current_limit = 10
    
    # try:
    #     # Initial run
    #     # print(f"Starting initial processing run with download limit: {current_limit}...")
    #     main_processing_loop(args, frames_base_dir, species_names, existing_dirs, download_limit=current_limit)
        
    #     while True:
    #         current_limit += 10
            
    #         next_run_time = time.time() + interval_seconds
    #         print(f"\nNext processing run scheduled at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(next_run_time))}")
    #         print(f"Next download limit will be: {current_limit}")
            
    #         # Sleep until next run
    #         time.sleep(interval_seconds)
            
    #         print(f"\nStarting scheduled processing run at {time.strftime('%Y-%m-%d %H:%M:%S')} with download limit: {current_limit}")
    #         main_processing_loop(args, frames_base_dir, species_names, existing_dirs, download_limit=current_limit)
            
    # except KeyboardInterrupt:
    #     print("\nProcess terminated by user")
    # except Exception as e:
    #     print(f"Error in main loop: {str(e)}")
    #     raise

