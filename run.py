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
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from picamera2 import Picamera2
from models.object_detection_utils import ObjectDetectionUtils
from models.detection import run_inference
from models.classification import infer_image
from models.insect_tracker import InsectTracker
from sensing_garden_client import SensingGardenClient

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
        self.device_id = "test_new_pipeline"
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
            # --- VALIDATION STEP ---
            if not any([payload.get("family"), payload.get("genus"), payload.get("species")]):
                print(f"  ✗ Skipping upload for detection at {payload['timestamp']}: No valid classification found.")
                failed_detections.append(payload)
                continue
            
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
                    bounding_box=payload.get('bbox'), # The key is 'bbox' in storage, argument is 'bounding_box'
                    track_id=payload.get('track_id')
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

    def shutdown(self):
        """Shutdown the processor, uploading any remaining detections."""
        print("\nShutting down processor...")
        if self.enable_uploads and self.local_detections:
            print("Uploading remaining detections before exit...")
            self.upload_local_batch()
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
        
        bgr_frame = frame.copy()
        
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
        track_ids = []
        if show_boxes:
            print(f"Frame {self.frame_count}: Sending {len(valid_detections)} detections to tracker")
            print(f"Current tracker has {len(self.tracker.current_tracks)} existing tracks")
        
        track_ids = self.tracker.update(valid_detections, self.frame_count)
        
        if show_boxes:
            print(f"Tracker returned IDs: {track_ids}")
            print(f"Tracker now has {len(self.tracker.current_tracks)} active tracks")
            print(f"Next track ID will be: {self.tracker.next_track_id}")
        
        # Process each detection with its track ID
        for i, det_data in enumerate(valid_detection_data):
            x, y, x2, y2 = det_data['x'], det_data['y'], det_data['x2'], det_data['y2']
            confidence = det_data['confidence']
            track_id = track_ids[i] if i < len(track_ids) else None
            
            if show_boxes:  # Only show in real-time mode
                print(f"Processing detection {i+1} with confidence {confidence:.3f}, track_id: {track_id}")
            
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
        
        # Summary message
        if not show_boxes and len(valid_detection_data) > 0:
            print(f"   📊 Found {len(valid_detection_data)} detection(s) above confidence threshold")
        elif show_boxes:
            print(f"Processed {len(valid_detection_data)} valid detections (confidence >= {self.confidence_threshold})")
        elif not show_boxes:
            print("   📊 No detections found")
        
        return bgr_frame

def initialize_camera():
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1080, 1080)})
    picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(1)
    return picam2

def run_realtime(enable_uploads=False, display=True, upload_interval=60):
    processor = InferenceProcessor(enable_uploads=enable_uploads)
    picam2 = initialize_camera()
    
    frame_count = 0
    last_upload_time = time.time()
    print("Starting real-time inference...")
    
    if enable_uploads:
        print(f"Database uploads are ENABLED. Batch uploads will occur every {upload_interval} seconds.")
    else:
        print("NOTE: Database uploads are DISABLED. Use --enable-uploads to enable them.")

    if not display:
        print("Display is DISABLED (headless mode).")
    
    try:
        while True:
            # Check for batch upload period
            if enable_uploads and (time.time() - last_upload_time >= upload_interval):
                print(f"\n--- Upload interval of {upload_interval}s reached. Pausing detections to upload batch. ---")
                processor.upload_local_batch()
                last_upload_time = time.time()
                print("--- Resuming detections. ---")

            frame = picam2.capture_array()
            
            if frame is None or frame.size == 0:
                continue
            
            processed_frame = processor.process_frame(frame, show_boxes=display)

            if display:
                cv2.imshow("Real-time Inference", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
            
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
    
    args = parser.parse_args()
    
    run_realtime(enable_uploads=args.enable_uploads, display=args.display, upload_interval=args.upload_interval)

if __name__ == "__main__":
    main()
