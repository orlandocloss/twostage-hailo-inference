import time
import os
import threading
import queue
import argparse
import random
import logging
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Import our refactored components
from main.record_video import VideoRecorder
from main.inference_from_video import VideoInferenceProcessor, process_video
from sensing_garden_client import SensingGardenClient

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContinuousPipeline:
    def __init__(self, video_dir="recordings", recording_duration=300, upload_percentage=10, 
                 device_id="pipeline", fps=15, resolution=(640, 640), enable_uploads=True,
                 confidence_threshold=0.35, class_names_path="data/london_invertebrates.txt"):
        """
        Initialize the continuous pipeline using imported components.
        """
        self.video_dir = Path(video_dir)
        self.upload_percentage = upload_percentage
        self.device_id = device_id
        self.enable_uploads = enable_uploads
        self.confidence_threshold = confidence_threshold
        
        # Threading controls
        self.stop_event = threading.Event()
        self.video_queue = queue.Queue()
        
        # --- Centralized Taxonomy Initialization ---
        logger.info("Initializing taxonomy...")
        try:
            self.class_names = self.load_class_names(class_names_path)
            self.families, self.genera, self.genus_to_family, self.species_to_genus = self.build_taxonomy()
            logger.info("✅ Taxonomy initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize taxonomy: {e}. Exiting.")
            raise
        
        # Initialize components
        self.recorder = VideoRecorder(
            output_dir=str(self.video_dir),
            fps=fps,
            resolution=resolution,
            recording_duration=recording_duration,
            device_id=self.device_id,
            video_queue=self.video_queue
        )
        self.sgc = None
        if self.enable_uploads:
            self.sgc = self.initialize_sensing_garden_client()
        
        # Threads
        self.recorder_thread = None
        self.processor_thread = None
        
        logger.info("🚀 Continuous Pipeline initialized")

    def load_class_names(self, class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(class_names)} class names")
        return class_names
    
    def build_taxonomy(self):
        if not self.class_names:
            return [], [], {}, {}
        taxonomy = self.get_taxonomy_from_gbif(self.class_names)
        families = taxonomy[1]
        genus_to_family = taxonomy[2]
        species_to_genus = taxonomy[3]
        genera = list(genus_to_family.keys())
        logger.info(f"Built taxonomy: {len(families)} families, {len(genera)} genera")
        return families, genera, genus_to_family, species_to_genus
    
    def get_taxonomy_from_gbif(self, species_list):
        taxonomy = {1: [], 2: {}, 3: {}}
        logger.info(f"\nBuilding taxonomy for {len(species_list)} species from GBIF...")
        for species_name in species_list:
            url = f"https://api.gbif.org/v1/species/match?name={species_name}&verbose=true"
            response = requests.get(url)
            data = response.json()
            if data.get('status') in ['ACCEPTED', 'SYNONYM']:
                family, genus = data.get('family'), data.get('genus')
                if family and genus:
                    if family not in taxonomy[1]: taxonomy[1].append(family)
                    taxonomy[2][genus] = family
                    taxonomy[3][species_name] = genus
                else: raise RuntimeError(f"{species_name} - missing family/genus data")
            else: raise RuntimeError(f"{species_name} not found in GBIF")
        taxonomy[1] = sorted(list(set(taxonomy[1])))
        return taxonomy

    def initialize_sensing_garden_client(self):
        try:
            return SensingGardenClient(
                base_url=os.environ.get('API_BASE_URL'),
                api_key=os.environ.get('SENSING_GARDEN_API_KEY'),
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                aws_region=os.environ.get('AWS_REGION', 'us-east-1')
            )
        except Exception as e:
            logger.error(f"Failed to initialize SensingGardenClient: {e}")
            return None

    def processor_worker(self):
        """Worker thread to process videos from the queue."""
        logger.info("🔍 Starting video processor worker...")
        
        while not self.stop_event.is_set() or not self.video_queue.empty():
            try:
                video_path = self.video_queue.get(timeout=1)
                
                logger.info(f"--- Processing Video: {video_path.name} ---")
                
                self.run_inference_on_video(video_path)
                if self.should_upload_video():
                    self.upload_video_file(video_path)
                self.delete_video(video_path)
                
                self.video_queue.task_done()
                
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue
            except Exception as e:
                logger.error(f"❌ Processor worker error: {e}", exc_info=True)
        
        logger.info("🔍 Video processor worker stopped")

    def run_inference_on_video(self, video_path):
        """Run the full inference process on a video file."""
        logger.info(f"🧠 Running inference on {video_path.name}")
        try:
            processor = VideoInferenceProcessor(
                # Pass the pre-loaded taxonomy data
                class_names=self.class_names,
                families=self.families,
                genera=self.genera,
                genus_to_family=self.genus_to_family,
                species_to_genus=self.species_to_genus,
                # Pass other configs
                enable_uploads=self.enable_uploads,
                device_id=self.device_id,
                confidence_threshold=self.confidence_threshold
            )
            process_video(video_path=str(video_path), processor=processor)
            
            if self.enable_uploads:
                processor.upload_all_detections()

        except Exception as e:
            logger.error(f"Error during inference for {video_path.name}: {e}")

    def should_upload_video(self):
        """Determine if this video should be uploaded based on percentage."""
        return random.randint(1, 100) <= self.upload_percentage

    def upload_video_file(self, video_path):
        """Upload a raw video file, using the filename for the timestamp."""
        if not self.enable_uploads or not self.sgc:
            logger.warning(f"Uploads disabled, skipping video upload for {video_path.name}")
            return
        
        try:
            # --- Derive timestamp from filename ---
            timestamp_str = video_path.stem.replace(f"{self.device_id}_", "")
            video_datetime = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            iso_timestamp = video_datetime.isoformat()
            
            logger.info(f"📤 Uploading video file: {video_path.name} with timestamp {iso_timestamp}")
            with open(video_path, "rb") as f:
                video_data = f.read()
            
            self.sgc.videos.upload_video(
                device_id=self.device_id,
                timestamp=iso_timestamp,
                video_path_or_data=video_data,
                content_type="video/mp4"
            )
            logger.info(f"✅ Video file uploaded successfully: {video_path.name}")
        except Exception as e:
            logger.error(f"❌ Error uploading video file {video_path.name}: {e}")
            
    def delete_video(self, video_path):
        """Delete a video file after processing."""
        try:
            video_path.unlink()
            logger.info(f"🗑️ Deleted video: {video_path.name}")
        except Exception as e:
            logger.error(f"⚠️ Could not delete video {video_path.name}: {e}")

    def start(self):
        """Start the continuous pipeline."""
        logger.info("🚀 Starting Continuous Pipeline...")
        self.recorder_thread = threading.Thread(target=self.recorder.start_continuous_recording, daemon=True)
        self.processor_thread = threading.Thread(target=self.processor_worker, daemon=True)
        self.recorder_thread.start()
        self.processor_thread.start()
        logger.info("✅ All worker threads started")
        
    def stop(self):
        """Stop the continuous pipeline gracefully, ensuring all work is finished."""
        logger.info("🔄 Stopping pipeline... recorder will finish current segment.")
        
        # 1. Signal recorder to stop. It will finish its current segment and push it to the queue.
        self.recorder.stop()
        # 2. Wait for the recorder thread to finish its job.
        if self.recorder_thread and self.recorder_thread.is_alive():
            self.recorder_thread.join()
        logger.info("✅ Recorder thread has stopped.")
        
        # 3. Wait for the processor to clear the queue.
        logger.info("...waiting for processor to finish all remaining videos...")
        self.video_queue.join()
        
        # 4. Signal the processor worker to stop now that the queue is empty.
        self.stop_event.set()
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join()
        logger.info("✅ Processor thread has stopped.")
        
        logger.info("✅ Pipeline stopped gracefully.")

def main():
    parser = argparse.ArgumentParser(description='Continuous video recording and inference pipeline.')
    parser.add_argument('--video-dir', type=str, default='recordings', help='Directory to save and monitor videos.')
    parser.add_argument('--duration', type=int, default=300, help='Duration of each video segment in seconds.')
    parser.add_argument('--upload-percentage', type=int, default=10, help='Percentage of video *files* to upload (0-100).')
    parser.add_argument('--device-id', type=str, default='pipeline', help='Device identifier for uploads.')
    parser.add_argument('--fps', type=int, default=15, help='Recording frame rate.')
    parser.add_argument('--resolution', type=str, default='640x640', help='Recording resolution in WIDTHxHEIGHT format.')
    parser.add_argument('--confidence', type=float, default=0.35, help='Confidence threshold for detections.')
    parser.add_argument('--disable-uploads', action='store_true', help='Disable all database uploads (detections and videos).')
    
    args = parser.parse_args()
    
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        logger.error(f"Error: Invalid resolution format '{args.resolution}'. Use WIDTHxHEIGHT.")
        return 1
    
    pipeline = ContinuousPipeline(
        video_dir=args.video_dir,
        recording_duration=args.duration,
        upload_percentage=args.upload_percentage,
        device_id=args.device_id,
        fps=args.fps,
        resolution=resolution,
        enable_uploads=not args.disable_uploads,
        confidence_threshold=args.confidence
    )
    
    try:
        pipeline.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown initiated by user.")
    except Exception as e:
        logger.error(f"💥 Fatal error in main loop: {e}", exc_info=True)
    finally:
        pipeline.stop()
    
    return 0

if __name__ == "__main__":
    exit(main())
