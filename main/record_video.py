import cv2
import time
import os
import threading
import queue
import logging
from datetime import datetime
from pathlib import Path
from picamera2 import Picamera2

logger = logging.getLogger(__name__)

class VideoRecorder:
    def __init__(self, output_dir="recordings", fps=15, resolution=(640, 640), 
                 recording_duration=300, device_id="recorder", video_queue=None):
        """
        Initialize the video recorder for gapless recording.
        
        Args:
            output_dir: Directory to save video files
            fps: Frames per second for recording
            resolution: Video resolution (width, height)
            recording_duration: Duration of each video segment in seconds
            device_id: Device identifier for filename
            video_queue: A queue to put the finished video file paths into.
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.resolution = resolution
        self.recording_duration = recording_duration
        self.device_id = device_id
        self.video_queue = video_queue
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.picam2 = None
        self.stop_event = threading.Event()
        
        # Frame queue for decoupling capture from writing
        self.frame_queue = queue.Queue(maxsize=fps * 5) # Buffer 5 seconds of frames
        self.frame_grabber_thread = None
        
        logger.info("Video Recorder component initialized.")
        logger.info(f"  - FPS: {self.fps}, Resolution: {self.resolution}")
    
    def initialize_camera(self):
        """Initialize and configure the camera."""
        try:
            self.picam2 = Picamera2()
            camera_config = self.picam2.create_video_configuration(
                main={"format": 'RGB888', "size": self.resolution}
            )
            self.picam2.configure(camera_config)
            self.picam2.set_controls({
                "FrameRate": float(self.fps),
                "AfMode": 0, "LensPosition": 0.0,
            })
            self.picam2.start()
            logger.info("Camera initializing...")
            time.sleep(2)
            logger.info("Camera ready.")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}", exc_info=True)
            raise

    def _frame_grabber(self):
        """Continuously grabs frames from the camera and puts them in a queue."""
        logger.info("📹 Frame grabber thread started.")
        while not self.stop_event.is_set():
            try:
                frame = self.picam2.capture_array()
                self.frame_queue.put(frame)
            except Exception:
                if not self.stop_event.is_set():
                    logger.exception("Error in frame grabber thread.")
                break
        logger.info("📹 Frame grabber thread stopped.")

    def record_segment(self, output_path):
        """Record a single video segment by consuming frames from the queue."""
        logger.info(f"🎬 Recording new segment: {output_path.name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, self.resolution)
        
        if not out.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < self.recording_duration:
                try:
                    frame = self.frame_queue.get(timeout=1)
                    out.write(frame)
                    frame_count += 1
                except queue.Empty:
                    if self.stop_event.is_set():
                        logger.info("Stop event received, finishing current segment early.")
                        break
                    continue
        finally:
            out.release()
            
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"  ✅ Segment complete: {output_path.name} ({frame_count} frames, {file_size_mb:.1f} MB)")
        
        if self.video_queue:
            self.video_queue.put(output_path)
    
    def start_continuous_recording(self):
        """Start continuous recording loop. This is intended to be the target of a thread."""
        logger.info("🚀 Starting continuous recording loop...")
        try:
            self.initialize_camera()
            self.frame_grabber_thread = threading.Thread(target=self._frame_grabber, daemon=True)
            self.frame_grabber_thread.start()
            
            while not self.stop_event.is_set():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.device_id}_{timestamp}.mp4"
                output_path = self.output_dir / filename
                
                self.record_segment(output_path)
        
        except Exception as e:
             if not self.stop_event.is_set():
                logger.error(f"❌ Fatal error in recording loop: {e}", exc_info=True)
        finally:
             logger.info("🎥 Recording loop has finished.")

    def stop(self):
        """Signals all internal loops to stop and cleans up resources."""
        logger.info("🔄 Stopping recorder component...")
        self.stop_event.set()
        
        if self.frame_grabber_thread and self.frame_grabber_thread.is_alive():
            self.frame_grabber_thread.join(timeout=2)
            
        if self.picam2:
            try:
                self.picam2.stop()
                logger.info("📸 Camera stopped")
            except Exception as e:
                logger.warning(f"Warning: Error stopping camera: {e}")
        
        logger.info("✅ Recorder component stopped.") 