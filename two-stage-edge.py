import cv2
import time
import os
import sys
import requests
import logging
import numpy as np
import argparse
import glob
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from picamera2 import Picamera2
from object_detection_utils import ObjectDetectionUtils
from detection import run_inference
from classification import infer_image
from sensing_garden_client import SensingGardenClient

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceProcessor:
    def __init__(self, model_path="small-generic.hef", labels_path="labels.txt", 
                 classification_model="london_141-multitask.hef", class_names_path="london_invertebrates.txt", 
                 batch_size=1, confidence_threshold=0.35):
        
        self.model_path = model_path
        self.labels_path = labels_path
        self.classification_model = classification_model
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.device_id = "test_edge_images"
        self.model_id = "london_141"
        
        self.class_names = self.load_class_names(class_names_path)
        self.families, self.genera, self.genus_to_family, self.species_to_genus = self.build_taxonomy()
        self.det_utils = ObjectDetectionUtils(labels_path)
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
    
    def upload_detection(self, frame, detection_data, timestamp):
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = buffer.tobytes()
            
            self.sgc.classifications.add(
                device_id=self.device_id,
                model_id=self.model_id,
                image_data=image_data,
                family=detection_data["family"],
                genus=detection_data["genus"],
                species=detection_data["species"],
                family_confidence=detection_data["family_confidence"],
                genus_confidence=detection_data["genus_confidence"],
                species_confidence=detection_data["species_confidence"],
                timestamp=timestamp,
                bounding_box=detection_data["bbox"],
                track_id=detection_data["track_id"]
            )
            print("Successfully uploaded detection to database")
        except Exception as e:
            print(f"Error uploading detection: {e}")
    
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
        infer_results = run_inference(
            net=self.model_path,
            input=frame,
            batch_size=self.batch_size,
            labels=self.labels_path,
            save_stream_output=False
        )
        
        bgr_frame = frame.copy()
        
        if show_boxes and len(infer_results) > 0:
            height, width = frame.shape[:2]
            
            for detection in infer_results:
                if len(detection) != 5:
                    continue
                    
                y_min, x_min, y_max, x_max, confidence = detection
                
                if confidence < self.confidence_threshold:
                    continue
                
                x, y = int(x_min * width), int(y_min * height)
                x2, y2 = int(x_max * width), int(y_max * height)
                
                cv2.rectangle(bgr_frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bgr_frame, f"{confidence:.2f}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                x, y, x2, y2 = max(0, x), max(0, y), min(width, x2), min(height, y2)
                
                if x2 <= x or y2 <= y:
                    continue
                
                cropped_region = cv2.resize(frame[y:y2, x:x2], (224, 224))
                classification_results = infer_image(cropped_region, hef_path=self.classification_model)
                
                detection_data = {
                    "family": None, "genus": None, "species": None,
                    "family_confidence": None, "genus_confidence": None, "species_confidence": None,
                    "bbox": self.convert_bbox_to_normalized(x, y, x2, y2, width, height),
                    "track_id": None
                }
                
                detection_data = self.process_classification_results(classification_results, detection_data)
                self.draw_classification_labels(bgr_frame, x, y2, detection_data)
                
                timestamp = datetime.now().isoformat()
                self.upload_detection(bgr_frame, detection_data, timestamp)
        
        return bgr_frame

def initialize_camera():
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1080, 1080)})
    picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(1)
    return picam2

def run_realtime():
    processor = InferenceProcessor()
    picam2 = initialize_camera()
    
    frame_count = 0
    print("Starting real-time inference...")
    
    try:
        while True:
            frame = picam2.capture_array()
            
            if frame is None or frame.size == 0:
                continue
            
            processed_frame = processor.process_frame(frame)
            cv2.imshow("Real-time Inference", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        picam2.stop()

def run_directory(directory_path):
    processor = InferenceProcessor()
    
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))
        image_files.extend(glob.glob(os.path.join(directory_path, ext.upper())))
    
    if not image_files:
        print(f"No images found in {directory_path}")
        return
    
    print(f"Processing {len(image_files)} images from {directory_path}")
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Failed to load {image_path}")
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = processor.process_frame(frame_rgb, show_boxes=False)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Two-stage inference processor')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--realtime', action='store_true', help='Run real-time inference from camera')
    group.add_argument('--directory', type=str, help='Process images from directory')
    
    args = parser.parse_args()
    
    if args.realtime:
        run_realtime()
    elif args.directory:
        run_directory(args.directory)

if __name__ == "__main__":
    main()
