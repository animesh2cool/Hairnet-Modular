"""
Streaming Endpoints
Video feed and WebSocket connections
"""
import cv2
import time
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse

from ...core.config import VideoConfig, DirectoryConfig, HooterConfig, hooter_config

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state (will be injected from main.py)
detector = None
display_frame = None
display_frame_lock = None
detection_active = False
current_violations = []
cap = None


class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@router.get("/video_feed")
def video_feed():
    """Optimized video streaming endpoint"""
    
    def generate():
        global display_frame, display_frame_lock, detection_active, cap
        
        while True:
            # Try to get frame from display_frame (processed)
            if display_frame_lock:
                with display_frame_lock:
                    if display_frame is not None:
                        frame = display_frame.copy()
                    else:
                        # Fallback: read directly from camera
                        if cap and cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                time.sleep(0.03)
                                continue
                            frame = cv2.resize(frame, (VideoConfig.STREAM_WIDTH, VideoConfig.STREAM_HEIGHT))
                        else:
                            time.sleep(0.03)
                            continue
            else:
                # If no lock available, try direct camera read
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.03)
                        continue
                    frame = cv2.resize(frame, (VideoConfig.STREAM_WIDTH, VideoConfig.STREAM_HEIGHT))
                else:
                    time.sleep(0.03)
                    continue
            
            # Fast JPEG encoding
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, 85,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ]
            
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            if not success:
                time.sleep(0.03)
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(1.0 / VideoConfig.STREAM_FPS)
    
    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time stats"""
    global detector, detection_active, current_violations
    
    await manager.connect(websocket)
    
    try:
        while True:
            if detector and detection_active:
                # Calculate violation counts
                from ...services.detection_service import DirectoryConfig as DC
                violation_video_dir = detector.violation_video_dir
                violation_videos_count = len(list(violation_video_dir.glob("*.mp4"))) if violation_video_dir.exists() else 0
                face_files_count = len(list(detector.violation_faces_dir.glob("*.jpg"))) if detector.violation_faces_dir.exists() else 0
                
                current_violation_faces = len(detector.current_violation_saved_faces) if hasattr(detector, 'current_violation_saved_faces') else 0
                
                # Hooter timing
                hooter_elapsed = 0
                hooter_remaining = 0
                if detector.hooter_active:
                    hooter_elapsed = time.time() - detector.hooter_last_triggered
                    hooter_remaining = max(0, hooter_config.DURATION - hooter_elapsed)
                
                stats_data = {
                    "violation_count": violation_videos_count,
                    "violations": current_violations,
                    "violation_faces_count": detector.violation_face_save_count,
                    "actual_face_files": face_files_count,
                    "counts_synchronized": violation_videos_count == detector.violation_face_save_count,
                    "violation_videos_count": violation_videos_count,
                    "recording_violation_video": detector.recording_violation_video,
                    "known_faces_count": len(detector.known_embeddings) if detector.known_embeddings else 0,
                    "current_violation_saved_faces": current_violation_faces,
                    "timestamp": datetime.now().isoformat(),
                    "detection_active": detection_active,
                    "hooter_enabled": hooter_config.is_enabled(),
                    "hooter_active": detector.hooter_active,
                    "hooter_elapsed_time": round(hooter_elapsed, 1),
                    "hooter_remaining_time": round(hooter_remaining, 1),
                    "hooter_duration": hooter_config.DURATION
                }
            else:
                stats_data = {
                    "violation_count": 0,
                    "violations": [],
                    "detection_active": False,
                    "timestamp": datetime.now().isoformat()
                }
            
            await websocket.send_text(json.dumps(stats_data))
            await asyncio.sleep(1)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.get("/", response_class=HTMLResponse)
async def get_homepage():
    """Serve homepage"""
    html_file = DirectoryConfig.STATIC_DIR / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    
    # Fallback: Return basic HTML if index.html doesn't exist
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head><title>Hairnet Detection System</title></head>
    <body>
        <h1>Hairnet Detection System</h1>
        <img src="/video_feed" width="960" height="620">
        <p>Access API docs at <a href="/docs">/docs</a></p>
    </body>
    </html>
    """)


@router.get("/violation-image/{filename}")
async def get_violation_image(filename: str):
    """Get specific violation image"""
    global detector
    if not detector:
        return {"error": "Detector not initialized"}
    
    file_path = detector.output_dir / filename
    if file_path.exists() and file_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
        return FileResponse(file_path, media_type="image/jpeg")
    
    return {"error": "Image not found"}


@router.get("/violation-face-image/{filename}")
async def get_violation_face_image(filename: str):
    """Get specific violation face image"""
    global detector
    if not detector:
        return {"error": "Detector not initialized"}
    
    file_path = detector.violation_faces_dir / filename
    if file_path.exists() and file_path.suffix.lower() == '.jpg':
        return FileResponse(file_path, media_type="image/jpeg")
    
    return {"error": "Face image not found"}


@router.get("/violation-video/{filename}")
async def get_violation_video(filename: str):
    """Get specific violation video"""
    global detector
    if not detector:
        return {"error": "Detector not initialized"}
    
    file_path = detector.violation_video_dir / filename
    if file_path.exists() and file_path.suffix.lower() == '.mp4':
        return FileResponse(
            file_path,
            media_type="video/mp4",
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
    
    return {"error": "Video not found"}


# Module initialization function (called from main.py)
def init_streaming_globals(det, disp_frame, lock, active, violations, video_cap):
    """Initialize global variables from main app"""
    global detector, display_frame, display_frame_lock, detection_active, current_violations, cap
    detector = det
    display_frame = disp_frame
    display_frame_lock = lock
    detection_active = active
    current_violations = violations
    cap = video_cap