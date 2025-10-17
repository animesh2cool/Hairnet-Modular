"""
Control Endpoints
Start/Stop detection, reload faces, configuration
"""
import logging
from fastapi import APIRouter
from pydantic import BaseModel

from ...core.config import VideoConfig
from ...models.schemas import ConfigUpdateRequest, PerformanceOptimizationRequest

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state (injected from main.py)
detector = None
processor = None
detection_active = False
start_detection_func = None
stop_detection_func = None


@router.post("/start-detection")
async def start_detection():
    """Start detection system"""
    if start_detection_func:
        return await start_detection_func()
    return {"status": "error", "message": "Start function not initialized"}


@router.post("/stop-detection")
async def stop_detection():
    """Stop detection system"""
    if stop_detection_func:
        return await stop_detection_func()
    return {"status": "error", "message": "Stop function not initialized"}


@router.post("/reload-faces")
async def reload_faces():
    """Force reload all known faces"""
    global detector
    
    if not detector:
        return {"status": "error", "message": "Detector not initialized"}
    
    try:
        detector.load_known_faces_with_alignment()
        return {
            "status": "success",
            "message": "Faces reloaded successfully",
            "known_faces_count": len(detector.known_embeddings),
            "employees": [
                {"employee_id": emp_id, "employee_name": name}
                for emp_id, name in detector.known_names.items()
            ]
        }
    except Exception as e:
        logger.exception(f"Face reload failed: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/configure-face-detection")
async def configure_face_detection(config: ConfigUpdateRequest):
    """Configure face detection parameters"""
    global detector
    
    if not detector:
        return {"status": "error", "message": "Detector not initialized"}
    
    try:
        if config.min_visibility and 0.1 <= config.min_visibility <= 0.9:
            detector.min_face_visibility = config.min_visibility
        
        if config.min_face_area and 100 <= config.min_face_area <= 2000:
            detector.face_area_threshold = config.min_face_area
        
        if config.recognition_threshold and 0.5 <= config.recognition_threshold <= 0.95:
            from ...core.config import FaceRecognitionConfig
            FaceRecognitionConfig.FACE_RECOGNITION_THRESHOLD = config.recognition_threshold
        
        return {
            "status": "success",
            "message": "Configuration updated",
            "current_settings": {
                "min_face_visibility": detector.min_face_visibility * 100,
                "min_face_area_pixels": detector.face_area_threshold,
                "recognition_threshold": FaceRecognitionConfig.FACE_RECOGNITION_THRESHOLD
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/optimize-performance")
async def optimize_performance(request: PerformanceOptimizationRequest):
    """Dynamically adjust detection performance"""
    global detector
    
    detection_fps = max(1, min(15, request.detection_fps))
    
    VideoConfig.PROCESSING_FPS = detection_fps
    
    if detector:
        detector.detection_interval = 1.0 / VideoConfig.PROCESSING_FPS
    
    performance_level = "High" if detection_fps >= 10 else "Medium" if detection_fps >= 5 else "Low"
    
    return {
        "message": f"Detection FPS optimized to {VideoConfig.PROCESSING_FPS}",
        "detection_fps": VideoConfig.PROCESSING_FPS,
        "stream_fps": VideoConfig.STREAM_FPS,
        "performance_level": performance_level
    }


@router.post("/reset-violation-state")
async def reset_violation_state():
    """Reset current violation state"""
    global detector
    
    if not detector:
        return {"status": "error", "message": "Detector not initialized"}
    
    try:
        detector.reset_violation_state()
        return {
            "status": "success",
            "message": "Violation state reset",
            "current_violation_saved_faces": len(detector.current_violation_saved_faces)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Module initialization
def init_control_globals(det, proc, active, start_func, stop_func):
    """Initialize global variables"""
    global detector, processor, detection_active, start_detection_func, stop_detection_func
    detector = det
    processor = proc
    detection_active = active
    start_detection_func = start_func
    stop_detection_func = stop_func