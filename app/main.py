"""
FastAPI Application Entry Point
Creates and configures the FastAPI app with all routes
"""
import cv2
import time
import queue
import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .core.config import (
    DirectoryConfig, VideoConfig, ServerConfig,
    VideoRecordingConfig
)
from .core.database import create_violations_table
from .services.detection_service import OptimizedHairnetDetector
from .api.api import api_router

# Import endpoint modules for global initialization
from .api.endpoints import streaming, control, status, hooter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== GLOBAL STATE ==========
detector: Optional[OptimizedHairnetDetector] = None
processor: Optional['OptimizedProcessor'] = None
cap: Optional[cv2.VideoCapture] = None
reader_thread: Optional[threading.Thread] = None

# Threading and synchronization
detection_active = False
stop_event = threading.Event()
detection_queue = queue.Queue(maxsize=VideoConfig.MAX_QUEUE_SIZE)
display_frame = None
display_frame_lock = threading.Lock()
current_violations = []

# Video buffer for recording
video_frame_buffer = []
video_buffer_lock = threading.Lock()

# Directory paths (set during startup)
output_dir: Optional[Path] = None
violation_faces_dir: Optional[Path] = None
violation_video_dir: Optional[Path] = None


# ========== OPTIMIZED PROCESSOR ==========
class OptimizedProcessor:
    """Handles detection in separate thread"""
    
    def __init__(self, detector_instance):
        self.detector = detector_instance
        self.processing_active = False
        self.detection_thread = None
    
    def start(self):
        """Start detection thread"""
        if self.processing_active:
            return
        
        self.processing_active = True
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.detection_thread.start()
        logger.info("✅ Detection thread started")
    
    def stop(self):
        """Stop detection thread"""
        self.processing_active = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        logger.info("🛑 Detection thread stopped")
    
    def _detection_worker(self):
        """Detection worker thread"""
        global display_frame, current_violations, video_frame_buffer
        
        while self.processing_active and not stop_event.is_set():
            try:
                frame_data = detection_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            if frame_data is None:
                detection_queue.task_done()
                break
            
            frame, timestamp = frame_data
            
            try:
                if self.detector:
                    # Get video buffer for recording
                    with video_buffer_lock:
                        buffer_copy = video_frame_buffer.copy()
                    
                    # Run detection
                    processed_frame, violations, zone_active = self.detector.detect_zone_only(
                        frame, buffer_copy
                    )
                    current_violations = violations
                    
                    # Update display frame
                    with display_frame_lock:
                        display_frame = processed_frame.copy()
            
            except Exception as e:
                logger.exception(f"Detection worker error: {e}")
            finally:
                detection_queue.task_done()


# ========== VIDEO FRAME READER ==========
def optimized_frame_reader():
    """Read frames and manage video buffer"""
    global cap, display_frame, video_frame_buffer
    
    frame_count = 0
    last_stream_time = 0
    stream_interval = 1.0 / VideoConfig.STREAM_FPS
    
    while not stop_event.is_set() and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            time.sleep(0.1)
            continue
        
        current_time = time.time()
        frame_count += 1
        
        # Resize frame
        frame = cv2.resize(frame, (VideoConfig.STREAM_WIDTH, VideoConfig.STREAM_HEIGHT))
        
        # Maintain video buffer
        with video_buffer_lock:
            video_frame_buffer.append(frame.copy())
            if len(video_frame_buffer) > VideoRecordingConfig.MAX_BUFFER_FRAMES:
                video_frame_buffer.pop(0)
        
        # Add to detection queue
        if frame_count % VideoConfig.FRAME_SKIP == 0:
            try:
                detection_queue.put((frame.copy(), current_time), block=False)
            except queue.Full:
                try:
                    _ = detection_queue.get_nowait()
                    detection_queue.put((frame.copy(), current_time), block=False)
                except Exception:
                    pass
        
        # Update display frame
        if current_time - last_stream_time >= stream_interval:
            with display_frame_lock:
                if display_frame is None:
                    display_frame = frame.copy()
            last_stream_time = current_time
        
        time.sleep(0.005)


# ========== INITIALIZATION FUNCTIONS ==========
def initialize_detector():
    """Initialize the detector"""
    global detector, processor, output_dir, violation_faces_dir, violation_video_dir
    
    try:
        if detector is not None:
            logger.info("Detector already initialized")
            return True
        
        # Setup directories
        output_dir, violation_faces_dir, violation_video_dir = DirectoryConfig.setup_directories()
        
        detector = OptimizedHairnetDetector(output_dir, violation_faces_dir, violation_video_dir)
        processor = OptimizedProcessor(detector)
        
        logger.info("✅ Detector initialized")
        return True
    
    except Exception as e:
        logger.exception(f"❌ Detector initialization failed: {e}")
        return False


def initialize_video():
    """Initialize video capture"""
    global cap
    
    try:
        cap = cv2.VideoCapture(VideoConfig.RTSP_URL)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 25)
            logger.info(f"✅ RTSP stream opened: {VideoConfig.RTSP_URL}")
            return True
        else:
            logger.warning("❌ RTSP failed, trying webcam...")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info("✅ Webcam opened as fallback")
                return True
            else:
                logger.error("❌ No video source available")
                return False
    
    except Exception as e:
        logger.error(f"❌ Video initialization error: {e}")
        return False


# ========== CONTROL FUNCTIONS ==========
async def start_detection_internal():
    """Start detection system"""
    global reader_thread, stop_event, detector, processor, cap, detection_active
    
    if not detector or not processor:
        if not initialize_detector():
            return {"status": "error", "message": "Failed to initialize detector"}
    
    if not cap:
        if not initialize_video():
            return {"status": "error", "message": "Failed to initialize video"}
    
    if reader_thread and reader_thread.is_alive():
        return {"status": "Detection already running"}
    
    # Clear stop event
    stop_event.clear()
    detection_active = True
    
    # Start processor
    if processor:
        processor.start()
    
    # Start frame reader
    reader_thread = threading.Thread(target=optimized_frame_reader, daemon=True)
    reader_thread.start()
    
    logger.info("✅ Detection started")
    return {"status": "Detection started successfully"}


async def stop_detection_internal():
    """Stop detection system"""
    global detection_active, stop_event, reader_thread, cap, processor
    
    detection_active = False
    stop_event.set()
    
    # Stop processor
    if processor:
        processor.stop()
    
    # Join reader thread
    if reader_thread and reader_thread.is_alive():
        reader_thread.join(timeout=3.0)
    
    # Release capture
    if cap:
        try:
            cap.release()
        except Exception:
            pass
    
    logger.info("🛑 Detection stopped")
    return {"status": "Detection stopped successfully"}


# ========== FASTAPI APP CREATION ==========
def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title=ServerConfig.TITLE,
        version=ServerConfig.VERSION
    )
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=str(DirectoryConfig.STATIC_DIR)), name="static")
    
    # Include API router
    app.include_router(api_router)
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        logger.info("🚀 Hairnet Detection System starting...")
        
        # Initialize detector
        if initialize_detector():
            logger.info(f"✅ Detector initialized with {len(detector.known_embeddings)} known faces")
        else:
            logger.error("❌ Detector initialization failed")
            return
        
        # Initialize database
        if create_violations_table():
            logger.info("✅ Database initialized")
        else:
            logger.warning("⚠️ Database setup failed - CSV logging only")
        
        # Initialize video
        if initialize_video():
            logger.info("✅ Video source initialized")
        else:
            logger.error("❌ Video initialization failed")
            return
        
        # Initialize endpoint modules with global state
        streaming.init_streaming_globals(
            detector, display_frame, display_frame_lock,
            detection_active, current_violations, cap  # ADD cap here
        )
        control.init_control_globals(
            detector, processor, detection_active,
            start_detection_internal, stop_detection_internal
        )
        status.init_status_globals(
            detector, cap, detection_active, streaming.manager
        )
        hooter.init_hooter_globals(detector)
        
        logger.info("🌐 Access at: http://localhost:{ServerConfig.PORT}")
        logger.info("✅ Startup complete!")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        global detection_active, cap, detector
        
        detection_active = False
        stop_event.set()
        
        # Turn off hooter
        if detector and detector.hooter_active:
            detector.turn_off_hooter()
            if detector.hooter_auto_off_timer:
                detector.hooter_auto_off_timer.cancel()
        
        if processor:
            processor.stop()
        
        if cap:
            cap.release()
        
        logger.info("🛑 System stopped")
    
    return app


# Create app instance
app = create_app()