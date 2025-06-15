import cv2
import time
import os
import sys
import requests
import logging
import numpy as np
import argparse
import glob
import threading
import queue
import json
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from picamera2 import Picamera2
from models.object_detection_utils import ObjectDetectionUtils
from models.detection import run_inference
from models.classification import infer_image
from models.insect_tracker import InsectTracker
from sensing_garden_client import SensingGardenClient
import tempfile

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceProcessor:
    def __init__(self, model_path="weights/small-generic.hef", labels_path="data/labels.txt", 
                 classification_model="weights/london_141-multitask.hef", class_names_path="data/london_invertebrates.txt", 
                 batch_size=1, confidence_threshold=0.35, enable_uploads=True):
        
        self.model_path = model_path
        self.labels_path = labels_path
        self.classification_model = classification_model
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.device_id = "test_pipeline2"
        self.model_id = "london_141"
        self.enable_uploads = enable_uploads
        
        self.local_detections = []
        
        # Initialize tracker (will be set up when we know frame dimensions)
        self.tracker = None
        self.frame_count = 0
        
        self.class_names = self.load_class_names(class_names_path)
        self.families, self.genera, self.genus_to_family, self.species_to_genus = self.build_taxonomy()
        self.det_utils = ObjectDetectionUtils(labels_path)
        
        if self.enable_uploads:
            self.sgc = self.initialize_sensing_garden_client()
        
    def load_class_names(self, class_names_path):
        try:
            with open(class_names_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(class_names)} class names")
            return class_names
        except Exception as e:
            print(f"Error loading class names: {e}")
            return []
    
    def build_taxonomy(self):
        if not self.class_names:
            return [], [], {}, {}
            
        taxonomy = self.get_taxonomy_from_gbif(self.class_names)
        families = taxonomy[1]
        genus_to_family = taxonomy[2]
        species_to_genus = taxonomy[3]
        genera = list(genus_to_family.keys())
        
        print(f"Built taxonomy: {len(families)} families, {len(genera)} genera")
        return families, genera, genus_to_family, species_to_genus
    
    def get_taxonomy_from_gbif(self, species_list):
        taxonomy = {1: [], 2: {}, 3: {}}
        
        print(f"\nBuilding taxonomy for {len(species_list)} species...")
        for species_name in species_list:
            url = f"https://api.gbif.org/v1/species/match?name={species_name}&verbose=true"
            try:
                response = requests.get(url)
                data = response.json()
                
                if data.get('status') in ['ACCEPTED', 'SYNONYM']:
                    family = data.get('family')
                    genus = data.get('genus')
                    
                    if family and genus:
                        if family not in taxonomy[1]:
                            taxonomy[1].append(family)
                        taxonomy[2][genus] = family
                        taxonomy[3][species_name] = genus
                    else:
                        print(f"Error: {species_name} - missing family/genus data")
                        sys.exit(1)
                else:
                    print(f"Error: {species_name} not found in GBIF")
                    sys.exit(1)
            except Exception as e:
                print(f"Error retrieving {species_name}: {e}")
                sys.exit(1)
        
        taxonomy[1] = sorted(list(set(taxonomy[1])))
        return taxonomy
    
    def initialize_sensing_garden_client(self):
        return SensingGardenClient(
            base_url=os.environ.get('API_BASE_URL'),
            api_key=os.environ.get('SENSING_GARDEN_API_KEY'),
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            aws_region=os.environ.get('AWS_REGION', 'us-east-1')
        )
    
    def convert_bbox_to_normalized(self, x, y, x2, y2, width, height):
        x_center = (x + x2) / 2.0 / width
        y_center = (y + y2) / 2.0 / height
        norm_width = (x2 - x) / width
        norm_height = (y2 - y) / height
        return [x_center, y_center, norm_width, norm_height]
    
    def store_detection_locally(self, frame, detection_data, timestamp):
        """Encodes the image and stores the complete upload payload locally."""
        if not self.enable_uploads:
            print(f"📷 Detected (uploads disabled): {detection_data.get('species', 'N/A')}")
            sys.stdout.flush()
            return

        try:
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = buffer.tobytes()

            payload = {
                "device_id": self.device_id,
                "model_id": self.model_id,
                "image_data": image_data,
                "timestamp": timestamp,
                **detection_data  # Unpacks family, genus, species, confidences, bbox, track_id
            }

            self.local_detections.append(payload)
            print(f"💿 Locally stored detection: {payload.get('species', 'N/A')} (total stored: {len(self.local_detections)})")
            sys.stdout.flush()

        except Exception as e:
            print(f"Error storing detection locally: {e}")

    def upload_local_batch(self):
        """Uploads all locally stored detections and clears the list on success."""
        if not self.enable_uploads or not self.local_detections:
            if self.enable_uploads:
                print("No local detections to upload.")
            return

        print(f"\n--- ☁️ Starting batch upload of {len(self.local_detections)} detections ---")
        
        failed_detections = []
        
        for payload in self.local_detections:
           
            try:
                # Reconstruct the function call EXACTLY as it was in the old working script.
                # This ensures the argument order and keys are correct.
                self.sgc.classifications.add(
                    device_id=payload['device_id'],
                    model_id=payload['model_id'],
                    image_data=payload['image_data'],
                    family=payload.get("family"),
                    genus=payload.get("genus"),
                    species=payload.get("species"),
                    family_confidence=payload.get("family_confidence"),
                    genus_confidence=payload.get("genus_confidence"),
                    species_confidence=payload.get("species_confidence"),
                    timestamp=payload['timestamp'],
                    # The key is 'bbox' in storage, argument is 'bounding_box'. Send as a raw list.
                    bounding_box=payload.get('bbox'), 
                    # CRITICAL FIX: The API requires the track_id to be a string.
                    track_id=str(payload['track_id']) 
                )
                print(f"  ✓ Successfully uploaded detection: {payload.get('species', 'N/A')}")
                sys.stdout.flush()
            except Exception as e:
                print(f"  ✗ Error uploading detection: {e}")
                # --- ENHANCED DEBUGGING ---
                print("    Failed Payload Data:")
                # Create a copy without the bulky image_data for printing
                debug_payload = payload.copy()
                debug_payload['image_data'] = f"<... {len(debug_payload.get('image_data', b''))} bytes ...>"
                for key, value in debug_payload.items():
                    print(f"      - {key}: {value}")
                sys.stdout.flush()
                failed_detections.append(payload)
                
        success_count = len(self.local_detections) - len(failed_detections)
        fail_count = len(failed_detections)
        
        print(f"--- ☁️ Batch upload complete. {success_count} successful, {fail_count} failed. ---\n")
        
        if fail_count > 0:
            print(f"WARNING: {fail_count} uploads failed. Retrying them in the next batch.")
            self.local_detections = failed_detections
        else:
            print("All uploads successful. Clearing local detection cache.")
            self.local_detections.clear()

    def create_and_upload_sanity_video(self, video_frames, fps=15):
        """Creates a video from a list of frames and uploads it."""
        if not self.enable_uploads or not video_frames:
            print("Sanity video creation skipped: No frames provided or uploads disabled.")
            return

        print(f"\n--- 🎬 Starting sanity video creation from {len(video_frames)} frames ---")
        
        # Get frame dimensions from the first frame
        height, width, _ = video_frames[0].shape
        
        temp_video_path = None
        try:
            # Create a temporary file to write the video to
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_video_path = temp_video.name

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            for frame in video_frames:
                # The processed frames are already in BGR format (from process_frame), so write directly
                out.write(frame)
            out.release()
            print(f"Temporary video saved to {temp_video_path}")

            # Read the video data for upload
            with open(temp_video_path, "rb") as f:
                video_data = f.read()

            # Upload the video
            print("Uploading sanity video...")
            self.sgc.videos.upload_video(
                device_id=self.device_id,
                timestamp=datetime.now().isoformat(),
                video_path_or_data=video_data,
                content_type="video/mp4"
            )
            print("  ✓ Sanity video uploaded successfully.")

        except Exception as e:
            print(f"  ✗ Error creating or uploading sanity video: {e}")
        finally:
            # Clean up the temporary file
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
                print(f"Temporary video file {temp_video_path} deleted.")
        print("--- 🎬 Sanity video process complete ---")

    def shutdown(self):
        """Shutdown the processor, uploading any remaining detections."""
        print("\nShutting down processor...")
        if self.enable_uploads and self.local_detections:
            print("Uploading remaining detections before exit...")
            self.upload_local_batch()
            # Note: A final sanity video is not created on shutdown to keep it simple.
        print("Shutdown complete.")
    
    def process_classification_results(self, classification_results, detection_data):
        for stream_name, result in classification_results.items():
            max_prob = np.max(result)
            top_indices = np.where(result >= (max_prob - 0.01))[0]
            
            if result.shape == (141,):  # Species
                detection_data["species"] = " | ".join([
                    self.class_names[idx] if idx < len(self.class_names) else f"Species {idx}" 
                    for idx in top_indices
                ])
                detection_data["species_confidence"] = float(max_prob)
            elif result.shape == (114,):  # Genus
                detection_data["genus"] = " | ".join([
                    self.genera[idx] if idx < len(self.genera) else f"Genus {idx}" 
                    for idx in top_indices
                ])
                detection_data["genus_confidence"] = float(max_prob)
            elif result.shape == (40,):  # Family
                detection_data["family"] = " | ".join([
                    self.families[idx] if idx < len(self.families) else f"Family {idx}" 
                    for idx in top_indices
                ])
                detection_data["family_confidence"] = float(max_prob)
        
        return detection_data
    
    def draw_classification_labels(self, frame, x, y2, detection_data):
        if detection_data["species"]:
            cv2.putText(frame, f"Species: {detection_data['species']}", (x, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if detection_data["genus"]:
            cv2.putText(frame, f"Genus: {detection_data['genus']}", (x, y2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if detection_data["family"]:
            cv2.putText(frame, f"Family: {detection_data['family']}", (x, y2 + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    def process_frame(self, frame, show_boxes=True):
        # Increment frame counter
        self.frame_count += 1
        
        # Initialize tracker on first frame
        if self.tracker is None:
            height, width = frame.shape[:2]
            # Use default track memory (equals max_frames) - objects can be matched within full history window
            self.tracker = InsectTracker(height, width, max_frames=30, w_dist=0.7, w_area=0.3, 
                                       cost_threshold=0.8)
            print(f"Initialized tracker for {width}x{height} frame with intelligent track memory")
        
        infer_results = run_inference(
            net=self.model_path,
            input=frame,
            batch_size=self.batch_size,
            labels=self.labels_path,
            save_stream_output=False
        )
        
        # Only show detailed debug in real-time mode (when show_boxes=True)
        if show_boxes:
            print(f"Found {len(infer_results)} raw detections")
        
        # Only create BGR copy if we need to draw on it
        bgr_frame = frame.copy() if show_boxes else frame
        
        # First pass: collect all valid detections for tracking
        valid_detections = []
        valid_detection_data = []
        
        if len(infer_results) > 0:
            height, width = frame.shape[:2]
            
            for detection in infer_results:
                if len(detection) != 5:
                    continue
                    
                y_min, x_min, y_max, x_max, confidence = detection
                
                if confidence < self.confidence_threshold:
                    if show_boxes:  # Only show in real-time mode
                        print(f"Skipping detection with confidence {confidence:.3f} (threshold: {self.confidence_threshold})")
                    continue
                
                # Convert to pixel coordinates
                x, y = int(x_min * width), int(y_min * height)
                x2, y2 = int(x_max * width), int(y_max * height)
                
                # Clamp coordinates
                x, y, x2, y2 = max(0, x), max(0, y), min(width, x2), min(height, y2)
                
                if x2 <= x or y2 <= y:
                    if show_boxes:  # Only show in real-time mode
                        print(f"Invalid crop dimensions: ({x}, {y}, {x2}, {y2})")
                    continue
                
                # Store detection for tracking (x1, y1, x2, y2 format)
                valid_detections.append([x, y, x2, y2])
                valid_detection_data.append({
                    'detection': detection,
                    'x': x, 'y': y, 'x2': x2, 'y2': y2,
                    'confidence': confidence
                })
        
        # ALWAYS update tracker with detections (even if empty list)
        track_ids = self.tracker.update(valid_detections, self.frame_count)
        
        # Reduced verbose output for performance
        if show_boxes and len(valid_detections) > 0:
            print(f"Frame {self.frame_count}: {len(valid_detections)} detections → {len(self.tracker.current_tracks)} active tracks")
        
        # Process each detection with its track ID
        for i, det_data in enumerate(valid_detection_data):
            x, y, x2, y2 = det_data['x'], det_data['y'], det_data['x2'], det_data['y2']
            confidence = det_data['confidence']
            track_id = track_ids[i] if i < len(track_ids) else None
            
            # Removed per-detection verbose output for performance
            
            # Draw bounding box and track ID
            if show_boxes:
                cv2.rectangle(bgr_frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bgr_frame, f"{confidence:.2f}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Draw track ID
                if track_id is not None:
                    cv2.putText(bgr_frame, f"ID:{track_id}", (x, y - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Perform classification
            cropped_region = cv2.resize(frame[y:y2, x:x2], (224, 224))
            classification_results = infer_image(cropped_region, hef_path=self.classification_model)
            
            # This dictionary uses "bbox" as the key, consistent with the old working script's logic.
            detection_data = {
                "family": None, "genus": None, "species": None,
                "family_confidence": None, "genus_confidence": None, "species_confidence": None,
                "bbox": self.convert_bbox_to_normalized(x, y, x2, y2, width, height),
                "track_id": track_id
            }
            
            detection_data = self.process_classification_results(classification_results, detection_data)
            
            if show_boxes:
                self.draw_classification_labels(bgr_frame, x, y2, detection_data)
            
            timestamp = datetime.now().isoformat()
            self.store_detection_locally(bgr_frame, detection_data, timestamp)
        
        # Simplified summary message for performance
        if not show_boxes and len(valid_detection_data) > 0:
            print(f"   📊 Found {len(valid_detection_data)} detection(s)")
        elif not show_boxes and len(valid_detection_data) == 0 and self.frame_count % 30 == 0:
            print("   📊 No detections (30 frames)")
        
        return bgr_frame

def initialize_camera():
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1080, 1080)})
    picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(1)
    return picam2

def run_realtime(enable_uploads=False, display=True, upload_interval=60, 
                 sanity_video_percent=0):
    
    # The feature is enabled if the user provides a percentage greater than 0.
    enable_sanity_video = sanity_video_percent > 0

    processor = InferenceProcessor(enable_uploads=enable_uploads)
    picam2 = initialize_camera()
    
    # --- State variables for the main loop ---
    last_upload_time = time.time()
    
    # Variables for just-in-time video recording
    is_recording = False
    video_frames_buffer = []
    recording_start_time = -1
    recording_end_time = -1

    def schedule_next_recording():
        """Determines the time window for the next sanity video."""
        nonlocal recording_start_time, recording_end_time
        
        # Calculate the duration of the video clip in seconds
        clip_duration = upload_interval * (sanity_video_percent / 100.0)
        
        if clip_duration < 1:
            recording_start_time = -1 # No recording this interval
            print("Interval too short or percentage too low to schedule a video.")
            return

        # Randomly choose a start time for the clip within the next interval
        # The start time is an offset from the beginning of the interval
        max_start_offset = upload_interval - clip_duration
        start_offset = np.random.uniform(0, max_start_offset)
        
        # These are relative to the 'last_upload_time' which marks the start of the interval
        recording_start_time = start_offset
        recording_end_time = start_offset + clip_duration
        print(f"Next sanity video scheduled: {clip_duration:.2f}s clip, starting at ~{start_offset:.2f}s into the next interval.")

    # Schedule the first recording session
    if enable_sanity_video:
        schedule_next_recording()

    try:
        while True:
            current_time = time.time()
            time_since_last_upload = current_time - last_upload_time

            # --- Batch Upload Logic ---
            if enable_uploads and (time_since_last_upload >= upload_interval):
                print(f"\n--- Upload interval of {upload_interval}s reached. Pausing detections to upload batch. ---")
                processor.upload_local_batch()
                if enable_sanity_video and video_frames_buffer:
                    # Estimate FPS from the number of frames recorded and the clip duration
                    actual_clip_duration = recording_end_time - recording_start_time
                    estimated_fps = len(video_frames_buffer) / actual_clip_duration if actual_clip_duration > 0 else 10
                    processor.create_and_upload_sanity_video(video_frames_buffer, fps=estimated_fps)
                
                # Clear buffers and schedule the next cycle
                video_frames_buffer.clear()
                is_recording = False
                if enable_sanity_video:
                    schedule_next_recording()

                print("--- Resuming detections. ---")
                last_upload_time = time.time()
                processor.frame_count = 0

            frame = picam2.capture_array()
            
            if frame is None or frame.size == 0:
                continue

            # --- Just-in-Time Recording Logic (check BEFORE processing for efficiency) ---
            should_record_this_frame = False
            if enable_sanity_video and recording_start_time != -1:
                # Check if the current time is within the recording window
                # Time is measured as an offset from the start of the interval
                time_into_interval = current_time - last_upload_time
                
                if not is_recording and time_into_interval >= recording_start_time:
                    is_recording = True
                    print(f"\n🎬 Starting sanity video recording at {time_into_interval:.2f}s into interval...")
                
                if is_recording:
                    should_record_this_frame = True
                    if time_into_interval >= recording_end_time:
                        is_recording = False
                        # Mark as done so we don't record again this interval
                        recording_start_time = -1 
                        print(f"🎬 Finished recording. Captured {len(video_frames_buffer)} frames.\n")

            # Process frame (only with full visualization if recording or displaying)
            processed_frame = processor.process_frame(frame, show_boxes=(display or should_record_this_frame))
            
            # Record frame if needed
            if should_record_this_frame:
                video_frames_buffer.append(processed_frame.copy())

            if display:
                cv2.imshow("Real-time Inference", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        processor.shutdown()
        if display:
            cv2.destroyAllWindows()
        picam2.stop()

def main():
    parser = argparse.ArgumentParser(description='Real-time two-stage inference processor for insect tracking.')
    
    parser.add_argument('--enable-uploads', action='store_true', 
                       help='Enable database uploads. Disabled by default for performance.')
    parser.add_argument('--headless', action='store_false', dest='display',
                       help='Run in headless mode without displaying the video feed.')
    parser.add_argument('--upload-interval', type=int, default=60,
                        help='The interval in seconds for batch uploading detections.')
    parser.add_argument('--sanity-video-percent', type=int, default=0,
                        help='Enables sanity video. Set to a percentage of interval time (e.g., 10). Default: 0 (disabled).')
    
    args = parser.parse_args()
    
    run_realtime(
        enable_uploads=args.enable_uploads, 
        display=args.display, 
        upload_interval=args.upload_interval,
        sanity_video_percent=args.sanity_video_percent
    )

if __name__ == "__main__":
    main()
