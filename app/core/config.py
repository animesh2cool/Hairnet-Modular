"""
Centralized Configuration Management
Loads all settings from environment variables
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# ========== DEVICE CONFIGURATION ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== DATABASE CONFIGURATION ==========
class DatabaseConfig:
    HOST = os.getenv('DB_HOST', 'localhost')
    PORT = int(os.getenv('DB_PORT', 3306))
    USER = os.getenv('DB_USER')
    PASSWORD = os.getenv('DB_PASSWORD')
    DATABASE = os.getenv('DB_NAME')
    CHARSET = 'utf8mb4'
    
    @classmethod
    def get_config_dict(cls):
        return {
            'host': cls.HOST,
            'port': cls.PORT,
            'user': cls.USER,
            'password': cls.PASSWORD,
            'database': cls.DATABASE,
            'charset': cls.CHARSET
        }

# ========== RTSP & VIDEO CONFIGURATION ==========
class VideoConfig:
    RTSP_URL = os.getenv('RTSP_URL', 'rtsp://admin:password@192.168.30.196:554')
    
    # Stream settings
    STREAM_WIDTH = int(os.getenv('STREAM_WIDTH', 960))
    STREAM_HEIGHT = int(os.getenv('STREAM_HEIGHT', 620))
    
    # Detection settings
    DETECTION_WIDTH = int(os.getenv('DETECTION_WIDTH', 960))
    DETECTION_HEIGHT = int(os.getenv('DETECTION_HEIGHT', 620))
    
    # Performance settings
    FRAME_SKIP = int(os.getenv('FRAME_SKIP', 1))
    PROCESSING_FPS = int(os.getenv('PROCESSING_FPS', 20))
    STREAM_FPS = int(os.getenv('STREAM_FPS', 20))
    
    # Queue settings
    MAX_QUEUE_SIZE = 3

# ========== DETECTION CONFIGURATION ==========
class DetectionConfig:
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.6))
    VIOLATION_TIME_SEC = float(os.getenv('VIOLATION_TIME_SEC', 0.2))
    
    # Zone settings
    LINE_Y_MIN = int(os.getenv('LINE_Y_MIN', 180))
    LINE_Y_MAX = int(os.getenv('LINE_Y_MAX', 550))
    ZONE_MARGIN = int(os.getenv('ZONE_MARGIN', 5))
    
    # Face class IDs
    FACE_CLASS_IDS = [0]

# ========== FACE RECOGNITION CONFIGURATION ==========
class FaceRecognitionConfig:
    FACE_RECOGNITION_THRESHOLD = float(os.getenv('FACE_RECOGNITION_THRESHOLD', 0.6))
    FACE_DETECTION_CONFIDENCE = float(os.getenv('FACE_DETECTION_CONFIDENCE', 0.95))
    FACE_VISIBILITY_THRESHOLD = float(os.getenv('FACE_VISIBILITY_THRESHOLD', 0.8))
    
    # Face processing
    FACE_INPUT_SIZE = int(os.getenv('FACE_INPUT_SIZE', 112))
    FACE_ALIGN_SIZE = int(os.getenv('FACE_ALIGN_SIZE', 160))
    EYE_ALIGNMENT_THRESHOLD = float(os.getenv('EYE_ALIGNMENT_THRESHOLD', 5.0))
    FACE_AREA_THRESHOLD = 400  # Minimum face area in pixels
    
    # Model files
    FACE_EMBEDDINGS_FILE = "face_embeddings.pkl"

# ========== VIDEO RECORDING CONFIGURATION ==========
class VideoRecordingConfig:
    VIDEO_BUFFER_SECONDS = int(os.getenv('VIDEO_BUFFER_SECONDS', 3))
    VIDEO_POST_SECONDS = int(os.getenv('VIDEO_POST_SECONDS', 2))
    VIDEO_BUFFER_FPS = int(os.getenv('VIDEO_BUFFER_FPS', 20))
    MAX_BUFFER_FRAMES = VIDEO_BUFFER_SECONDS * VIDEO_BUFFER_FPS

# ========== HOOTER/ALARM CONFIGURATION ==========
class HooterConfig:
    ENABLED = os.getenv('HOOTER_ENABLED', 'True').lower() == 'true'
    IP = os.getenv('HOOTER_IP', '192.168.30.21')
    RELAY = int(os.getenv('HOOTER_RELAY', 4))
    DURATION = int(os.getenv('HOOTER_DURATION', 10))
    TIMEOUT = float(os.getenv('HOOTER_TIMEOUT', 5.0))
    
    # NEW: Add cooldown to prevent repeated triggers
    COOLDOWN_SECONDS = int(os.getenv('HOOTER_COOLDOWN', 30))  # 30 sec cooldown
    
    @property
    def ON_URL(self):
        return f"http://{self.IP}/80/07"
    
    @property
    def OFF_URL(self):
        return f"http://{self.IP}/80/06"

# ========== DIRECTORY CONFIGURATION ==========
class DirectoryConfig:
    OUTPUT_BASE_DIR = Path(os.getenv('OUTPUT_BASE_DIR', 'violations'))
    KNOWN_FACES_DIR = Path(os.getenv('KNOWN_FACES_DIR', 'known_faces'))
    STATIC_DIR = Path(os.getenv('STATIC_DIR', 'static'))
    
    @classmethod
    def setup_directories(cls):
        """Create all necessary directories"""
        from datetime import datetime
        
        # Create known faces directory
        cls.KNOWN_FACES_DIR.mkdir(exist_ok=True)
        
        # Create static directory
        cls.STATIC_DIR.mkdir(exist_ok=True)
        
        # Create timestamped output directory
        report_time = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = cls.OUTPUT_BASE_DIR / f"report_QA_Room_{report_time}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        violation_faces_dir = output_dir / "violation_faces"
        violation_faces_dir.mkdir(exist_ok=True)
        
        violation_video_dir = output_dir / "violation_videos"
        violation_video_dir.mkdir(exist_ok=True)
        
        return output_dir, violation_faces_dir, violation_video_dir

# ========== MODEL CONFIGURATION ==========
class ModelConfig:
    HAIRNET_MODEL_PATH = "runs/detect/train/weights/best.pt"
    FACE_MODEL_PATH = "yolov8m-face.pt"
    FACENET_PRETRAINED = "vggface2"

# ========== SERVER CONFIGURATION ==========
class ServerConfig:
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 4053))
    RELOAD = os.getenv('RELOAD', 'True').lower() == 'true'
    TITLE = "Hairnet Detection System"
    VERSION = "3.0.0"

# ========== INSTANTIATE HOOTER CONFIG ==========
hooter_config = HooterConfig()