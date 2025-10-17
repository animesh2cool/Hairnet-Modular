"""
Hooter Control Endpoints
Manual and automatic hooter/alarm control with master switch
"""
import time
import logging
from fastapi import APIRouter, HTTPException
from ...core.config import hooter_config
from ...models.schemas import HooterStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state (injected from main.py)
detector = None


@router.post("/on")
async def manual_hooter_on():
    """
    Manually trigger hooter ON
    Will fail if master switch is OFF
    """
    global detector
    
    if not detector:
        return {"status": "error", "message": "Detector not initialized"}
    
    # Check master switch
    if not hooter_config.is_enabled():
        return {
            "status": "blocked",
            "message": "Hooter master switch is OFF - Cannot activate",
            "master_switch": hooter_config.MASTER_SWITCH,
            "hint": "Set HOOTER_MASTER_SWITCH=ON in .env file to enable"
        }
    
    success = detector.trigger_hooter(duration=999999)  # Manual mode - no auto-off
    
    return {
        "status": "success" if success else "failed",
        "hooter_active": detector.hooter_active,
        "master_switch": hooter_config.MASTER_SWITCH,
        "message": "Hooter turned ON manually (will not auto-off)"
    }


@router.post("/off")
async def manual_hooter_off():
    """
    Manually turn hooter OFF
    Always works (safety feature) regardless of master switch
    """
    global detector
    
    if not detector:
        return {"status": "error", "message": "Detector not initialized"}
    
    success = detector.turn_off_hooter()
    
    if success:
        detector.hooter_active = False
        if detector.hooter_auto_off_timer:
            detector.hooter_auto_off_timer.cancel()
    
    return {
        "status": "success" if success else "failed",
        "hooter_active": detector.hooter_active,
        "message": "Hooter turned OFF manually"
    }


@router.post("/trigger")
async def trigger_hooter_timed(duration: int = None):
    """
    Trigger hooter for specified duration
    Will fail if master switch is OFF
    """
    global detector
    
    if not detector:
        return {"status": "error", "message": "Detector not initialized"}
    
    # Check master switch
    if not hooter_config.is_enabled():
        return {
            "status": "blocked",
            "message": "Hooter master switch is OFF - Cannot activate",
            "master_switch": hooter_config.MASTER_SWITCH,
            "hint": "Set HOOTER_MASTER_SWITCH=ON in .env file to enable"
        }
    
    duration = duration or hooter_config.DURATION
    
    if duration < 1 or duration > 60:
        return {"error": "Duration must be between 1 and 60 seconds"}
    
    success = detector.trigger_hooter(duration)
    
    return {
        "status": "success" if success else "failed",
        "hooter_active": detector.hooter_active if detector else False,
        "master_switch": hooter_config.MASTER_SWITCH,
        "duration": duration,
        "message": f"Hooter will sound for {duration} seconds"
    }


@router.get("/master-switch")
async def get_master_switch_status():
    """
    Get master switch status
    Returns current state of the physical kill switch
    """
    return {
        "master_switch": hooter_config.MASTER_SWITCH,
        "enabled": hooter_config.is_enabled(),
        "status_message": hooter_config.get_status_message(),
        "description": "Master switch controls if hooter can be triggered at all",
        "how_to_change": "Edit HOOTER_MASTER_SWITCH in .env file (ON/OFF) and restart server"
    }


@router.post("/toggle")
async def toggle_hooter_enabled(enabled: bool):
    """
    DEPRECATED: Use master switch in .env instead
    This endpoint is kept for backwards compatibility
    """
    return {
        "status": "deprecated",
        "message": "Please use HOOTER_MASTER_SWITCH in .env file instead",
        "current_master_switch": hooter_config.MASTER_SWITCH,
        "master_switch_enabled": hooter_config.is_enabled(),
        "instruction": "Set HOOTER_MASTER_SWITCH=ON or HOOTER_MASTER_SWITCH=OFF in .env and restart"
    }


@router.get("/status", response_model=HooterStatus)
async def get_hooter_status():
    """Get current hooter status including master switch"""
    global detector
    
    if not detector:
        return {
            "hooter_enabled": hooter_config.is_enabled(),
            "hooter_active": False,
            "elapsed_time": 0,
            "remaining_time": 0,
            "duration": hooter_config.DURATION,
            "ip": hooter_config.IP,
            "relay": hooter_config.RELAY,
            "last_triggered": 0
        }
    
    elapsed_time = 0
    remaining_time = 0
    
    if detector.hooter_active:
        elapsed_time = time.time() - detector.hooter_last_triggered
        remaining_time = max(0, hooter_config.DURATION - elapsed_time)
    
    return {
        "hooter_enabled": hooter_config.is_enabled(),
        "hooter_active": detector.hooter_active,
        "elapsed_time": round(elapsed_time, 1),
        "remaining_time": round(remaining_time, 1),
        "duration": hooter_config.DURATION,
        "ip": hooter_config.IP,
        "relay": hooter_config.RELAY,
        "last_triggered": detector.hooter_last_triggered
    }


# Module initialization
def init_hooter_globals(det):
    """Initialize global variables"""
    global detector
    detector = det