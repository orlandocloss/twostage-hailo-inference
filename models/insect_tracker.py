# pip install scipy numpy

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

class BoundingBox:
    def __init__(self, x, y, width, height, frame_id, track_id=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.area = width * height
        self.frame_id = frame_id
        self.track_id = track_id
    
    def center(self):
        return (self.x + self.width/2, self.y + self.height/2)
    
    @classmethod
    def from_xyxy(cls, x1, y1, x2, y2, frame_id, track_id=None):
        """Create BoundingBox from x1,y1,x2,y2 coordinates"""
        width = x2 - x1
        height = y2 - y1
        return cls(x1, y1, width, height, frame_id, track_id)

class InsectTracker:
    def __init__(self, image_height, image_width, max_frames=30, w_dist=0.7, w_area=0.3, cost_threshold=0.8):
        self.image_height = image_height
        self.image_width = image_width
        self.max_dist = np.sqrt(image_height**2 + image_width**2)
        self.max_frames = max_frames
        self.w_dist = w_dist
        self.w_area = w_area
        self.cost_threshold = cost_threshold
        
        self.tracking_history = deque(maxlen=max_frames)
        self.current_tracks = []
        self.next_track_id = 0
    
    def calculate_cost(self, box1, box2):
        """Calculate cost between two bounding boxes as per equation (4)"""
        # Calculate center points
        cx1, cy1 = box1.center()
        cx2, cy2 = box2.center()
        
        # Euclidean distance (equation 1)
        dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        
        # Normalized distance (equation 2 used for normalization)
        norm_dist = dist / self.max_dist
        
        # Area cost (equation 3)
        min_area = min(box1.area, box2.area)
        max_area = max(box1.area, box2.area)
        area_cost = min_area / max_area if max_area > 0 else 1.0
        
        # Final cost (equation 4)
        cost = (norm_dist * self.w_dist) + ((1 - area_cost) * self.w_area)
        
        return cost
    
    def build_cost_matrix(self, prev_boxes, curr_boxes):
        """Build cost matrix for Hungarian algorithm"""
        n_prev = len(prev_boxes)
        n_curr = len(curr_boxes)
        n = max(n_prev, n_curr)
        
        # Initialize cost matrix with high values
        cost_matrix = np.ones((n, n)) * 999.0
        
        # Fill in actual costs
        for i in range(n_prev):
            for j in range(n_curr):
                cost_matrix[i, j] = self.calculate_cost(prev_boxes[i], curr_boxes[j])
        
        return cost_matrix, n_prev, n_curr
    
    def update(self, new_detections, frame_id):
        """
        Update tracking with new detections from YOLO
        
        Args:
            new_detections: List of YOLO detection boxes (x1, y1, x2, y2 format)
            frame_id: Current frame number
            
        Returns:
            List of track IDs corresponding to each detection
        """
        # Convert YOLO detections to BoundingBox objects
        new_boxes = []
        for i, detection in enumerate(new_detections):
            x1, y1, x2, y2 = detection[:4]
            bbox = BoundingBox.from_xyxy(x1, y1, x2, y2, frame_id)
            new_boxes.append(bbox)
        
        # If this is the first frame, assign new track IDs to all boxes
        if not self.current_tracks:
            track_ids = []
            for box in new_boxes:
                box.track_id = self.next_track_id
                track_ids.append(self.next_track_id)
                self.next_track_id += 1
            self.current_tracks = new_boxes
            self.tracking_history.append(new_boxes)
            return track_ids
        
        # Build cost matrix
        cost_matrix, n_prev, n_curr = self.build_cost_matrix(self.current_tracks, new_boxes)
        
        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Assign track IDs based on the matching
        assigned_curr_indices = set()
        track_ids = [None] * len(new_boxes)
        
        for i, j in zip(row_indices, col_indices):
            # Only consider valid assignments (not dummy rows/columns)
            if i < n_prev and j < n_curr:
                # Check if cost is below threshold
                if cost_matrix[i, j] < self.cost_threshold:
                    # Assign the track ID from previous box to current box
                    new_boxes[j].track_id = self.current_tracks[i].track_id
                    track_ids[j] = self.current_tracks[i].track_id
                    assigned_curr_indices.add(j)
        
        # Assign new track IDs to unassigned current boxes (new insects)
        for j in range(n_curr):
            if j not in assigned_curr_indices:
                new_boxes[j].track_id = self.next_track_id
                track_ids[j] = self.next_track_id
                self.next_track_id += 1
        
        # Update current tracks
        self.current_tracks = new_boxes
        
        # Add to tracking history
        self.tracking_history.append(new_boxes)
        
        return track_ids 