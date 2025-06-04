import cv2
import time
import os
import sys
import requests
import logging
import numpy as np
from datetime import datetime
from picamera2 import Picamera2
from object_detection_utils import ObjectDetectionUtils
from detection import run_inference
from classification import infer_image  # Import the classification function

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleInference:
    def __init__(self, model_path="small-generic.hef", labels_path="labels.txt", classification_model="london_141-multitask.hef", class_names_path="london_invertebrates.txt", batch_size=1, save_results=False, show_boxes=True):
        self.model_path = model_path
        self.labels_path = labels_path
        self.classification_model = classification_model
        self.batch_size = batch_size
        self.save_results = save_results
        self.show_boxes = show_boxes
        
        # Load class names
        self.class_names = []
        try:
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.class_names)} class names from {class_names_path}")
        except Exception as e:
            print(f"Error loading class names: {e}")
        
        # Get taxonomy information for the species
        if self.class_names:
            taxonomy = self.get_taxonomy(self.class_names)
            self.families = taxonomy[1]  # List of families
            self.genus_to_family = taxonomy[2]  # Dictionary mapping genus to family
            self.species_to_genus = taxonomy[3]  # Dictionary mapping species to genus
            
            # Create genus list from the dictionary keys
            self.genera = list(self.genus_to_family.keys())
            
            print(f"Extracted {len(self.families)} families and {len(self.genera)} genera")
        else:
            self.families = []
            self.genera = []
            self.genus_to_family = {}
            self.species_to_genus = {}
        
        # Create output directory for saving results
        if self.save_results:
            self.output_dir = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Saving results to {self.output_dir}")
        
        # Initialize camera
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1080, 1080)})
        self.picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
        self.picam2.configure(camera_config)
        self.picam2.start()
        
        # Initialize detection utils
        self.det_utils = ObjectDetectionUtils(labels_path)
        
        # Wait a moment for initialization
        time.sleep(1)
        
    def get_taxonomy(self, species_list):
        """
        Retrieves taxonomic information for a list of species from GBIF API.
        Creates a hierarchical taxonomy dictionary with family, genus, and species relationships.
        """
        taxonomy = {1: [], 2: {}, 3: {}}
        species_to_genus = {}
        genus_to_family = {}
        
        logger.info(f"Building taxonomy from GBIF for {len(species_list)} species")
        
        print("\nTaxonomy Results:")
        print("-" * 80)
        print(f"{'Species':<30} {'Family':<20} {'Genus':<20} {'Status'}")
        print("-" * 80)
        
        for species_name in species_list:
            url = f"https://api.gbif.org/v1/species/match?name={species_name}&verbose=true"
            try:
                response = requests.get(url)
                data = response.json()
                
                if data.get('status') == 'ACCEPTED' or data.get('status') == 'SYNONYM':
                    family = data.get('family')
                    genus = data.get('genus')
                    
                    if family and genus:
                        status = "OK"
                        
                        print(f"{species_name:<30} {family:<20} {genus:<20} {status}")
                        
                        species_to_genus[species_name] = genus
                        genus_to_family[genus] = family
                        
                        if family not in taxonomy[1]:
                            taxonomy[1].append(family)
                        
                        taxonomy[2][genus] = family
                        taxonomy[3][species_name] = genus
                    else:
                        error_msg = f"Species '{species_name}' found in GBIF but family and genus not found, could be spelling error in species, check GBIF"
                        logger.error(error_msg)
                        print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                        print(f"Error: {error_msg}")
                        sys.exit(1)  # Stop the script
                else:
                    error_msg = f"Species '{species_name}' not found in GBIF, could be spelling error, check GBIF"
                    logger.error(error_msg)
                    print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                    print(f"Error: {error_msg}")
                    sys.exit(1)  # Stop the script
                    
            except Exception as e:
                error_msg = f"Error retrieving data for species '{species_name}': {str(e)}"
                logger.error(error_msg)
                print(f"{species_name:<30} {'Error':<20} {'Error':<20} FAILED")
                print(f"Error: {error_msg}")
                sys.exit(1)  # Stop the script
        
        taxonomy[1] = sorted(list(set(taxonomy[1])))
        print("-" * 80)
        
        num_families = len(taxonomy[1])
        num_genera = len(taxonomy[2])
        num_species = len(taxonomy[3])
        
        print("\nFamily indices:")
        for i, family in enumerate(taxonomy[1]):
            print(f"  {i}: {family}")
        
        print("\nGenus indices:")
        for i, genus in enumerate(taxonomy[2].keys()):
            print(f"  {i}: {genus}")
        
        print("\nSpecies indices:")
        for i, species in enumerate(species_list):
            print(f"  {i}: {species}")
        
        logger.info(f"Taxonomy built: {num_families} families, {num_genera} genera, {num_species} species")
        return taxonomy
        
    def process_frame(self, frame):
        # Get inference results using run_inference
        infer_results = run_inference(
            net=self.model_path,
            input=frame,  # Pass the frame directly as a numpy array
            batch_size=self.batch_size,
            labels=self.labels_path,
            save_stream_output=False
        )
        
        # Print the inference results
        print("Inference Results:")
        for i, result in enumerate(infer_results):
            print(f"Result {i}:")
            print(result)
            
        # Make a copy of the original frame to avoid modification
        #bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr_frame = frame.copy()
        
        # Draw bounding boxes if enabled
        if self.show_boxes and len(infer_results) > 0:
            height, width = frame.shape[:2]
            for detection in infer_results:
                # Extract bounding box coordinates (format is [y_min, x_min, y_max, x_max, confidence])
                if len(detection) == 5:  # Ensure we have the expected format
                    y_min, x_min, y_max, x_max, confidence = detection
                    
                    # Convert normalized coordinates to pixel coordinates
                    x = int(x_min * width)
                    y = int(y_min * height)
                    x2 = int(x_max * width)
                    y2 = int(y_max * height)
                    
                    # Draw rectangle
                    cv2.rectangle(bgr_frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    
                    # Add confidence text
                    confidence_text = f"{confidence:.2f}"
                    cv2.putText(bgr_frame, confidence_text, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Crop region from original frame for classification
                    try:
                        # Ensure coordinates are within bounds
                        x = max(0, x)
                        y = max(0, y)
                        x2 = min(width, x2)
                        y2 = min(height, y2)
                        
                        if x2 <= x or y2 <= y:
                            print(f"Invalid crop dimensions: ({x}, {y}, {x2}, {y2})")
                            continue
                        
                        # Crop the detected region
                        cropped_region = frame[y:y2, x:x2]
                        
                        # Resize to expected input size (224x224 is common, adjust as needed)
                        cropped_region_resized = cv2.resize(cropped_region, (224, 224))
                        
                        # Run classification on the cropped region
                        print(f"\n--- Classifying object at coordinates: ({x}, {y}, {x2}, {y2}) ---")
                        classification_results = infer_image(cropped_region_resized, hef_path=self.classification_model)
                        
                        # Print classification results and draw class name with highest probability
                        for stream_name, result in classification_results.items():
                            print(f"Output stream: {stream_name}")
                            print(f"Shape: {result.shape}")
                            print(f"Results (probabilities): {result}")
                            
                            # Check if this is the species output stream (shape should be (141,))
                            if result.shape == (141,):
                                # Get top classes
                                top_classes = np.argsort(result)[::-1][:5]  # Top 5 classes
                                print(f"Top 5 classes: {top_classes}")
                                print(f"Top 5 probabilities: {result[top_classes]}")
                                
                                # Get highest probability and find all classes with equally high probabilities
                                max_prob = np.max(result)
                                tolerance = 0.01  # Tolerance for considering probabilities as equal
                                top_indices = np.where(result >= (max_prob - tolerance))[0]
                                
                                print(f"Highest probability species indices: {top_indices} with probabilities: {result[top_indices]}")
                                
                                # Create label with all top species
                                species_labels = []
                                for idx in top_indices:
                                    species_name = f"Species {idx}"
                                    if self.class_names and idx < len(self.class_names):
                                        species_name = self.class_names[idx]
                                    species_labels.append(f"{species_name}({result[idx]:.2f})")
                                
                                # Draw species labels on the image
                                label = " | ".join(species_labels)
                                cv2.putText(bgr_frame, label, (x, y2 + 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            # Check if this is the genus output stream (shape should be (114,))
                            elif result.shape == (114,):
                                # Get highest probability and find all genera with equally high probabilities
                                max_prob = np.max(result)
                                tolerance = 0.01  # Tolerance for considering probabilities as equal
                                top_indices = np.where(result >= (max_prob - tolerance))[0]
                                
                                print(f"Highest probability genus indices: {top_indices} with probabilities: {result[top_indices]}")
                                
                                # Create label with all top genera
                                genus_labels = []
                                for idx in top_indices:
                                    genus_name = f"Genus {idx}"
                                    if self.genera and idx < len(self.genera):
                                        genus_name = self.genera[idx]
                                    genus_labels.append(f"{genus_name}({result[idx]:.2f})")
                                
                                # Draw genus labels on the image
                                label = " | ".join(genus_labels)
                                cv2.putText(bgr_frame, label, (x, y2 + 40), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            
                            # Check if this is the family output stream (shape should be (40,))
                            elif result.shape == (40,):
                                # Get highest probability and find all families with equally high probabilities
                                max_prob = np.max(result)
                                tolerance = 0.01  # Tolerance for considering probabilities as equal
                                top_indices = np.where(result >= (max_prob - tolerance))[0]
                                
                                print(f"Highest probability family indices: {top_indices} with probabilities: {result[top_indices]}")
                                
                                # Create label with all top families
                                family_labels = []
                                for idx in top_indices:
                                    family_name = f"Family {idx}"
                                    if self.families and idx < len(self.families):
                                        family_name = self.families[idx]
                                    family_labels.append(f"{family_name}({result[idx]:.2f})")
                                
                                # Draw family name and probability on the imageSSS
                                label = " | ".join(family_labels)
                                cv2.putText(bgr_frame, label, (x, y2 + 60), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                    except Exception as e:
                        print(f"Error in classification: {e}")
                        import traceback
                        traceback.print_exc()
        
        return bgr_frame
        
    def run(self):
        frame_count = 0
        
        try:
            while True:
                print(f"Processing frame {frame_count}")
                
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Verify frame is valid
                if frame is None or frame.size == 0:
                    print("WARNING: Captured an empty frame")
                    time.sleep(0.1)
                    continue
                    
                print(f"Captured frame shape: {frame.shape}, dtype: {frame.dtype}")
                
                try:
                    # Process the frame
                    processed_frame = self.process_frame(frame)
                    
                    # Save results if enabled
                    if self.save_results and processed_frame is not None:
                        output_path = os.path.join(self.output_dir, f"frame_{frame_count:04d}.jpg")
                        save_success = cv2.imwrite(output_path, processed_frame)
                        print(f"Saved frame to {output_path}, success: {save_success}")
                    
                    # Display the frame
                    cv2.imshow("Inference", processed_frame)
                    
                except Exception as e:
                    print(f"Error during inference: {e}")
                    import traceback
                    traceback.print_exc()
                    # Use original frame if processing fails
                    if frame is not None and frame.size > 0:
                        # Convert to BGR for display
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Inference", bgr_frame)
                
                # Exit on 'q' press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit key pressed")
                    break
                
                frame_count += 1
                # Small delay to ensure UI updates
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            print("Cleaning up")
            cv2.destroyAllWindows()
            self.picam2.stop()

if __name__ == "__main__":
    inference = SimpleInference(save_results=True, show_boxes=True)
    inference.run()
