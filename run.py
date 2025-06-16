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
                 batch_size=1, confidence_threshold=0.35, enable_uploads=True, device_id="test_pipeline2"):
        
        self.model_path = model_path
        self.labels_path = labels_path
        self.classification_model = classification_model
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.device_id = device_id
        self.model_id = "london_141"
        self.enable_uploads = enable_uploads
        
        # Double buffer for asynchronous uploads
        self.detection_buffers = {'A': [], 'B': []}
        self.active_buffer_key = 'A'
        self.buffer_lock = threading.Lock()
        
        # Initialize tracker (will be set up when we know frame dimensions)
        self.tracker = None
        self.frame_count = 0
        
        self.class_names = self.load_class_names(class_names_path)
        self.families, self.genera, self.genus_to_family, self.species_to_genus = self.build_taxonomy()
        self.det_utils = ObjectDetectionUtils(labels_path)
        
        if self.enable_uploads:
            self.sgc = self.initialize_sensing_garden_client()
        
    def get_full_buffer_and_swap(self):
        """
        Safely swaps the active detection buffer and returns the one that is full.
        This is designed to be called by the main thread, which then passes the
        returned buffer to the upload worker.
        """
        with self.buffer_lock:
            full_buffer = self.detection_buffers[self.active_buffer_key]
            self.detection_buffers[self.active_buffer_key] = []  # Clear the buffer for future use
            self.active_buffer_key = 'B' if self.active_buffer_key == 'A' else 'A'
            return full_buffer

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
            
            with self.buffer_lock:
                self.detection_buffers[self.active_buffer_key].append(payload)
            print(f"💿 Locally stored detection: {payload.get('species', 'N/A')} (buffer '{self.active_buffer_key}' size: {len(self.detection_buffers[self.active_buffer_key])})")
            sys.stdout.flush()

        except Exception as e:
            print(f"Error storing detection locally: {e}")

    def upload_local_batch(self, detections_to_upload):
        """
        Uploads a given batch of detections. Returns a list of failed detections.
        This is designed to be called from the uploader worker thread.
        """
        if not self.enable_uploads or not detections_to_upload:
            if self.enable_uploads:
                print("No local detections to upload.")
            return []

        print(f"\n--- ☁️ Starting batch upload of {len(detections_to_upload)} detections ---")
        
        failed_detections = []
        
        for payload in detections_to_upload:
           
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
                
        success_count = len(detections_to_upload) - len(failed_detections)
        fail_count = len(failed_detections)
        
        print(f"--- ☁️ Batch upload complete. {success_count} successful, {fail_count} failed. ---\n")
        
        if fail_count > 0:
            print(f"WARNING: {fail_count} uploads failed. They will be re-queued for the next upload cycle.")
        else:
            print("All uploads successful. Clearing local detection cache.")
        
        return failed_detections

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
        # The main loop now handles final uploads before calling shutdown.
        # We can add any other final cleanup here if needed in the future.
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
            # OPTIMIZATION: Store the small cropped region, not the entire large frame.
            # This dramatically reduces CPU usage in the main processing thread.
            self.store_detection_locally(cropped_region, detection_data, timestamp)
        
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

def uploader_worker(upload_queue, processor):
    """A worker function to run in a separate thread, handling uploads."""
    failed_detections_to_retry = []
    print("Uploader worker started.")

    while True:
        try:
            item = upload_queue.get()
            if item is None:  # Sentinel value to signal thread to exit
                print("Uploader worker received exit signal.")
                # Before exiting, try one last time to upload any failed items
                if failed_detections_to_retry:
                    print(f"Attempting final upload for {len(failed_detections_to_retry)} previously failed detections.")
                    processor.upload_local_batch(failed_detections_to_retry)
                break

            detections, video_frames, video_fps = item
            
            # Prepend failed detections from the previous run to the current batch
            if failed_detections_to_retry:
                print(f"Retrying {len(failed_detections_to_retry)} failed uploads from previous batch.")
                detections = failed_detections_to_retry + detections
                failed_detections_to_retry.clear()

            # 1. Upload detections if any exist
            if detections:
                failed_detections_to_retry = processor.upload_local_batch(detections)
            
            # 2. Upload video if any frames were captured
            if video_frames:
                processor.create_and_upload_sanity_video(video_frames, fps=video_fps)

        except Exception as e:
            logger.error(f"Unhandled exception in uploader thread: {e}", exc_info=True)
        finally:
            # This is crucial to allow queue.join() to unblock
            upload_queue.task_done()
            
    print("Uploader worker finished.")

def run_realtime(enable_uploads=False, display=True, upload_interval=60, 
                 sanity_video_percent=0, device_id="test_pipeline2"):
    
    # The feature is enabled if the user provides a percentage greater than 0.
    enable_sanity_video = sanity_video_percent > 0

    processor = InferenceProcessor(enable_uploads=enable_uploads, device_id=device_id)
    picam2 = initialize_camera()

    # --- Asynchronous Uploader Setup ---
    upload_queue = None
    uploader_thread = None
    if enable_uploads:
        upload_queue = queue.Queue()
        uploader_thread = threading.Thread(
            target=uploader_worker, 
            args=(upload_queue, processor), 
            daemon=True
        )
        uploader_thread.start()
    
    # --- State variables for the main loop ---
    last_upload_time = time.time()
    
    # --- Double buffer for just-in-time video recording ---
    video_buffers = {'A': [], 'B': []}
    active_video_buffer_key = 'A'
    
    is_recording = False
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

            # --- Asynchronous Batch Upload Logic ---
            if enable_uploads and (time_since_last_upload >= upload_interval):
                print(f"\n--- Upload interval of {upload_interval}s reached. Offloading data for async upload. ---")
                
                # 1. Get the full detection buffer from the processor and swap to the new one
                detections_to_upload = processor.get_full_buffer_and_swap()
                
                # 2. Swap video buffers and get the full one
                video_to_upload = video_buffers[active_video_buffer_key]
                video_buffers[active_video_buffer_key] = []  # Clear the now-inactive buffer
                active_video_buffer_key = 'B' if active_video_buffer_key == 'A' else 'A'

                target_fps = 15 # Default
                if enable_sanity_video and video_to_upload:
                    # --- REVISED FPS & DURATION LOGIC ---
                    # 1. Calculate the actual average FPS achieved during the last interval.
                    # This gives a realistic playback speed based on system performance.
                    interval_duration = time.time() - last_upload_time
                    actual_fps = processor.frame_count / interval_duration if interval_duration > 0 else 15
                    # Clamp to a reasonable range for stability.
                    target_fps = max(5, min(actual_fps, 30))

                    # 2. Calculate the target number of frames for the desired video length.
                    clip_duration = upload_interval * (sanity_video_percent / 100.0)
                    target_frame_count = int(clip_duration * target_fps)

                    # 3. If processing was slow and we have too few frames, pad the video.
                    # This ensures the video is always the desired length.
                    num_missing_frames = target_frame_count - len(video_to_upload)
                    if num_missing_frames > 0:
                        print(f"⚠️ Processing was slow. Padding video with {num_missing_frames} frames to meet target duration.")
                        last_frame = video_to_upload[-1]
                        video_to_upload.extend([last_frame] * num_missing_frames)
                    
                    # Ensure we don't have too many frames (can happen if FPS fluctuates)
                    video_to_upload = video_to_upload[:target_frame_count]

                    final_video_duration = len(video_to_upload) / target_fps
                    print(f"📹 Assembling video: {len(video_to_upload)} frames from a {clip_duration:.1f}s window.")
                    print(f"📹 Target playback: {target_fps:.1f} FPS for a {final_video_duration:.1f}s video.")

                # 3. Queue the data for the uploader thread
                if detections_to_upload or video_to_upload:
                    print(f"Queuing {len(detections_to_upload)} detections and {len(video_to_upload)} video frames for upload.")
                    upload_queue.put((detections_to_upload, video_to_upload, target_fps))
                else:
                    print("No new detections or video frames to upload in this interval.")

                # 4. Reset timers and schedule the next recording cycle
                last_upload_time = time.time()
                processor.frame_count = 0  # Reset frame count for the new interval
                is_recording = False
                if enable_sanity_video:
                    schedule_next_recording()

                print("--- Resuming detections into new buffers. ---")

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
                        print(f"🎬 Finished recording. Captured {len(video_buffers[active_video_buffer_key])} frames.\n")

            # NEW: Record the raw frame for the sanity video BEFORE processing.
            # This is fast and decouples video recording from processing time,
            # ensuring consistent video length and frame rate.
            if should_record_this_frame:
                video_buffers[active_video_buffer_key].append(frame.copy())

            # Now, process the frame for detections. This is the slow part.
            # The returned annotated frame is only used for the live display.
            if display:
                processed_frame = processor.process_frame(frame, show_boxes=True)
                cv2.imshow("Real-time Inference", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # In headless mode, we still need to process for detections, but don't need the annotated frame.
                processor.process_frame(frame, show_boxes=False)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # --- Graceful Shutdown ---
        print("\n--- Initiating shutdown sequence... ---")
        
        # 1. Stop hardware and UI first to prevent crashes on Ctrl+C.
        print("Stopping camera...")
        picam2.stop()
        if display:
            print("Closing display windows...")
            cv2.destroyAllWindows()

        # 2. Handle final uploads now that hardware is safe.
        if enable_uploads and upload_queue is not None and uploader_thread.is_alive():
            print("Queueing final batch of data before exit...")
            # Get any remaining data from the currently active buffers
            final_detections = processor.get_full_buffer_and_swap()
            final_video = video_buffers[active_video_buffer_key]
            
            if final_detections or final_video:
                # Use default FPS for final video chunk
                upload_queue.put((final_detections, final_video, 15))
            
            # Signal the uploader to finish its queue and exit
            print("Waiting for all pending uploads to complete...")
            upload_queue.put(None)  # Sentinel to stop the worker
            upload_queue.join()     # Wait for all items to be processed
            uploader_thread.join()  # Wait for the thread to terminate
            print("All uploads finished.")

        processor.shutdown()
        print("--- Shutdown complete. ---")

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
    parser.add_argument('--device-id', type=str, default='test_pipeline2',
                        help='Device ID for uploads and identification. Default: test_pipeline2')
    
    args = parser.parse_args()
    
    run_realtime(
        enable_uploads=args.enable_uploads, 
        display=args.display, 
        upload_interval=args.upload_interval,
        sanity_video_percent=args.sanity_video_percent,
        device_id=args.device_id
    )

if __name__ == "__main__":
    main()
