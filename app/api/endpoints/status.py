"""
Status Endpoints
Health checks, logs, statistics, and system information
"""
import csv
import time
import logging
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter

from ...core.config import (
    VideoConfig, DirectoryConfig, FaceRecognitionConfig,
    VideoRecordingConfig, hooter_config, ServerConfig
)
from ...core.database import (
    get_violations_from_database,
    check_database_health as db_health_check
)
from ...models.schemas import SystemStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state (injected from main.py)
detector = None
cap = None
detection_active = False
websocket_manager = None


@router.get("/health")
async def health_check():
    """Comprehensive health check"""
    global detector, cap, detection_active, websocket_manager
    
    violation_faces_count = detector.violation_face_save_count if detector else 0
    violation_videos_count = 0
    known_faces_count = 0
    
    if detector:
        if detector.violation_video_dir.exists():
            violation_videos_count = len(list(detector.violation_video_dir.glob("*.mp4")))
        known_faces_count = len(detector.known_embeddings) if detector.known_embeddings else 0
    
    elapsed_time = 0
    remaining_time = 0
    if detector and detector.hooter_active:
        elapsed_time = time.time() - detector.hooter_last_triggered
        remaining_time = max(0, hooter_config.DURATION - elapsed_time)
    
    return {
        "status": "healthy",
        "detection_active": detection_active,
        "detector_initialized": detector is not None,
        "video_initialized": cap is not None and cap.isOpened() if cap else False,
        "websocket_connections": len(websocket_manager.active_connections) if websocket_manager else 0,
        "violation_count": violation_videos_count,
        "internal_violation_counter": detector.violation_count if detector else 0,
        "violation_faces_saved": violation_faces_count,
        "violation_videos_saved": violation_videos_count,
        "known_faces_count": known_faces_count,
        "face_recognition_enabled": detector.mtcnn is not None if detector else False,
        "currently_recording_video": detector.recording_violation_video if detector else False,
        "duplicate_prevention": {
            "active": hasattr(detector, 'current_violation_saved_faces') if detector else False,
            "description": "Prevents saving multiple faces of same person per violation"
        },
        "face_visibility_check": {
            "active": hasattr(detector, 'min_face_visibility') if detector else False,
            "min_visibility_percent": detector.min_face_visibility * 100 if detector else 0,
            "description": "Only recognizes faces with >70% visibility"
        },
        "performance_config": {
            "stream_resolution": f"{VideoConfig.STREAM_WIDTH}x{VideoConfig.STREAM_HEIGHT}",
            "detection_resolution": f"{VideoConfig.DETECTION_WIDTH}x{VideoConfig.DETECTION_HEIGHT}",
            "stream_fps": VideoConfig.STREAM_FPS,
            "detection_fps": VideoConfig.PROCESSING_FPS,
            "frame_skip": VideoConfig.FRAME_SKIP,
            "video_buffer_seconds": VideoRecordingConfig.VIDEO_BUFFER_SECONDS,
            "video_post_seconds": VideoRecordingConfig.VIDEO_POST_SECONDS,
            "face_recognition_threshold": FaceRecognitionConfig.FACE_RECOGNITION_THRESHOLD,
            "face_visibility_threshold": detector.min_face_visibility * 100 if detector else 0,
            "min_face_area_pixels": detector.face_area_threshold if detector else 0
        },
        "hooter_status": {
            "enabled": hooter_config.ENABLED,
            "active": detector.hooter_active if detector else False,
            "duration": hooter_config.DURATION,
            "elapsed_time": round(elapsed_time, 1),
            "remaining_time": round(remaining_time, 1),
            "ip": hooter_config.IP,
            "relay_number": hooter_config.RELAY
        },
        "face_alignment": {
            "enabled": True,
            "face_input_size": f"{FaceRecognitionConfig.FACE_INPUT_SIZE}x{FaceRecognitionConfig.FACE_INPUT_SIZE}",
            "similarity_metric": "cosine_similarity",
            "alignment_method": "eye_based",
            "model": "FaceNet_InceptionResnetV1"
        },
        "timestamp": datetime.now().isoformat(),
        "video_based_violation_counting": True
    }


@router.get("/status")
async def get_status():
    """Get system status"""
    global detector, detection_active
    
    violations_count = 0
    if detector and detector.violation_video_dir.exists():
        violations_count = len(list(detector.violation_video_dir.glob("*.mp4")))
    
    return {
        "status": "active" if detection_active else "inactive",
        "violations_count": violations_count,
        "detector_initialized": detector is not None
    }


@router.get("/violation-logs")
async def get_violation_logs():
    """Read violation logs from CSV"""
    global detector
    
    if not detector:
        return {"violations": [], "total": 0}
    
    try:
        violations = []
        log_file = detector.log_file
        
        if log_file and log_file.exists():
            with open(log_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    violations.append(row)
        
        # Sort by timestamp (newest first)
        violations.sort(key=lambda x: x.get('Timestamp', ''), reverse=True)
        
        return {
            "violations": violations,
            "total": len(violations)
        }
    except Exception as e:
        logger.exception(f"Error reading violation logs: {e}")
        return {"violations": [], "total": 0}


@router.get("/violation-videos")
async def get_violation_videos():
    """Get list of saved violation videos"""
    global detector
    
    if not detector:
        return {"violation_videos": []}
    
    violation_video_files = []
    if detector.violation_video_dir.exists():
        for video_file in detector.violation_video_dir.glob("*.mp4"):
            violation_video_files.append({
                "filename": video_file.name,
                "timestamp": video_file.stat().st_mtime,
                "size": video_file.stat().st_size,
                "size_mb": round(video_file.stat().st_size / (1024*1024), 2),
                "path": str(video_file)
            })
    
    return {
        "violation_videos": sorted(
            violation_video_files,
            key=lambda x: x["timestamp"],
            reverse=True
        )
    }


@router.get("/violation-faces")
async def get_violation_faces():
    """Get list of saved violation face images"""
    global detector
    
    if not detector:
        return {"violation_faces": []}
    
    violation_face_files = []
    if detector.violation_faces_dir.exists():
        for face_file in detector.violation_faces_dir.glob("*.jpg"):
            violation_face_files.append({
                "filename": face_file.name,
                "timestamp": face_file.stat().st_mtime,
                "size": face_file.stat().st_size,
                "path": str(face_file)
            })
    
    return {
        "violation_faces": sorted(
            violation_face_files,
            key=lambda x: x["timestamp"],
            reverse=True
        )
    }


@router.get("/violations-database")
async def get_violations_database(limit: int = 50, offset: int = 0):
    """Get violations from database with pagination"""
    return get_violations_from_database(limit, offset)


@router.get("/database-health")
async def check_database_health():
    """Check database connection and table status"""
    return db_health_check()


@router.get("/known-faces")
async def get_known_faces():
    """Get list of known faces"""
    global detector
    
    if detector and detector.known_embeddings:
        faces_info = []
        for emp_id, emp_name in detector.known_names.items():
            faces_info.append({
                "employee_id": emp_id,
                "employee_name": emp_name,
                "has_embedding": emp_id in detector.known_embeddings
            })
        return {
            "known_faces": faces_info,
            "total_count": len(faces_info)
        }
    else:
        return {"known_faces": [], "total_count": 0}


@router.get("/face-folders-status")
async def get_face_folders_status():
    """Get detailed status of known_faces folders"""
    folders_status = []
    total_images = 0
    
    if DirectoryConfig.KNOWN_FACES_DIR.exists():
        for person_folder in DirectoryConfig.KNOWN_FACES_DIR.iterdir():
            if not person_folder.is_dir():
                continue
            
            folder_name = person_folder.name
            
            # Parse folder name
            try:
                parts = folder_name.split('_', 1)
                if len(parts) == 2:
                    employee_id = parts[0]
                    employee_name = parts[1].replace('_', ' ')
                else:
                    employee_id = folder_name
                    employee_name = folder_name
            except:
                employee_id = folder_name
                employee_name = folder_name
            
            # Count images
            image_files = []
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
                image_files.extend(list(person_folder.glob(ext)))
            
            image_count = len(image_files)
            total_images += image_count
            
            # Check if has embeddings
            has_embeddings = detector and employee_id in detector.known_embeddings if detector else False
            
            folders_status.append({
                "folder_name": folder_name,
                "employee_id": employee_id,
                "employee_name": employee_name,
                "image_count": image_count,
                "has_embeddings": has_embeddings,
                "folder_path": str(person_folder)
            })
    
    return {
        "folders": sorted(folders_status, key=lambda x: x["employee_id"]),
        "total_folders": len(folders_status),
        "total_images": total_images,
        "processed_employees": len(detector.known_embeddings) if detector else 0
    }


@router.get("/face-detection-stats")
async def get_face_detection_stats():
    """Get detailed face detection statistics"""
    global detector
    
    if not detector:
        return {"error": "Detector not initialized"}
    
    recognized_faces = 0
    unknown_faces = 0
    total_violations = 0
    
    if detector.log_file and detector.log_file.exists():
        try:
            with open(detector.log_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_violations += 1
                    emp_id = row.get('Employee_ID')
                    if emp_id and emp_id != 'Unknown':
                        recognized_faces += 1
                    else:
                        unknown_faces += 1
        except Exception as e:
            logger.error(f"Error reading face stats: {e}")
    
    return {
        "total_violations": total_violations,
        "recognized_faces": recognized_faces,
        "unknown_faces": unknown_faces,
        "recognition_rate": round((recognized_faces / total_violations * 100) if total_violations > 0 else 0, 2),
        "duplicate_prevention_active": hasattr(detector, 'current_violation_saved_faces'),
        "face_visibility_threshold": detector.min_face_visibility * 100 if detector else 0,
        "min_face_area_threshold": detector.face_area_threshold if detector else 0,
        "current_violation_saved_faces": len(detector.current_violation_saved_faces) if hasattr(detector, 'current_violation_saved_faces') else 0,
        "known_employees_count": len(detector.known_embeddings) if detector.known_embeddings else 0
    }


@router.get("/validation-check")
async def validation_check():
    """Check violation counts using video count as primary metric"""
    global detector
    
    if not detector:
        return {"error": "Detector not initialized"}
    
    violation_files_count = 0
    violation_videos_count = 0
    
    if detector.violation_faces_dir.exists():
        violation_files_count = len(list(detector.violation_faces_dir.glob("*.jpg")))
    
    if detector.violation_video_dir.exists():
        violation_videos_count = len(list(detector.violation_video_dir.glob("*.mp4")))
    
    # Count unique employees
    unique_employees = set()
    if detector.log_file and detector.log_file.exists():
        try:
            with open(detector.log_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    emp_id = row.get('Employee_ID')
                    if emp_id and emp_id != 'Unknown':
                        unique_employees.add(emp_id)
        except Exception as e:
            logger.error(f"Error reading log: {e}")
    
    return {
        "violation_count": violation_videos_count,
        "internal_violation_counter": detector.violation_count,
        "violation_face_save_count": detector.violation_face_save_count,
        "actual_violation_files": violation_files_count,
        "actual_violation_videos": violation_videos_count,
        "unique_employees_in_violations": len(unique_employees),
        "duplicate_prevention_active": hasattr(detector, 'current_violation_saved_faces'),
        "face_visibility_check_active": hasattr(detector, 'min_face_visibility'),
        "min_face_visibility_threshold": detector.min_face_visibility * 100 if detector else 0,
        "video_based_counting": True,
        "counts_match": violation_videos_count == detector.violation_face_save_count,
        "files_match_counter": violation_files_count == detector.violation_face_save_count,
        "status": "synced" if violation_videos_count == detector.violation_face_save_count else "mismatched"
    }


# Module initialization
def init_status_globals(det, video_cap, active, ws_manager):
    """Initialize global variables"""
    global detector, cap, detection_active, websocket_manager
    detector = det
    cap = video_cap
    detection_active = active
    websocket_manager = ws_manager