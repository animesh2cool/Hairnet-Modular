"""
Pydantic Models for Request/Response Validation
"""
from pydantic import BaseModel
from typing import Optional, List, Dict


class ViolationRecord(BaseModel):
    """Violation record model"""
    timestamp: str
    confidence: float
    image_path: str
    center_y: int


class SystemStatus(BaseModel):
    """System status model"""
    status: str
    violations_count: int
    current_frame: Optional[str] = None
    last_violation: Optional[ViolationRecord] = None


class DetectionResult(BaseModel):
    """Detection result model"""
    frame_base64: str
    violations: List[Dict]
    timestamp: str
    zone_active: bool


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    detection_active: bool
    detector_initialized: bool
    video_initialized: bool
    websocket_connections: int
    violation_faces_saved: int
    violation_videos_saved: int
    known_faces_count: int


class ViolationFaceData(BaseModel):
    """Face data from violation"""
    file_path: str
    employee_id: Optional[str]
    employee_name: str
    face_confidence: float


class HooterStatus(BaseModel):
    """Hooter status response"""
    hooter_enabled: bool
    hooter_active: bool
    elapsed_time: float
    remaining_time: float
    duration: int
    ip: str
    relay: int
    last_triggered: float


class ConfigUpdateRequest(BaseModel):
    """Face detection configuration update"""
    min_visibility: Optional[float] = None
    min_face_area: Optional[int] = None
    recognition_threshold: Optional[float] = None


class PerformanceOptimizationRequest(BaseModel):
    """Performance optimization request"""
    detection_fps: int