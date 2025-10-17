"""
Face Alignment and Processing Utilities
Handles face preprocessing, alignment, and embedding operations
"""
import math
import numpy as np
import cv2
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import logging

from ..core.config import FaceRecognitionConfig, DEVICE

logger = logging.getLogger(__name__)


def calculate_eye_angle(left_eye: tuple, right_eye: tuple) -> float:
    """
    Calculate angle between eyes for face alignment
    
    Args:
        left_eye: (x, y) coordinates of left eye
        right_eye: (x, y) coordinates of right eye
    
    Returns:
        Angle in degrees
    """
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.atan2(dy, dx) * 180.0 / math.pi
    return angle


def align_face_by_eyes(
    image: Image.Image,
    left_eye: tuple,
    right_eye: tuple,
    face_size: int = FaceRecognitionConfig.FACE_INPUT_SIZE
) -> Image.Image:
    """
    Align face based on eye positions and resize to consistent size
    
    Args:
        image: PIL Image or numpy array
        left_eye: (x, y) coordinates of left eye
        right_eye: (x, y) coordinates of right eye
        face_size: target face size (default 112x112)
    
    Returns:
        Aligned and resized face image as PIL Image
    """
    try:
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Calculate eye distance and angle
        eye_distance = np.sqrt(
            (right_eye[0] - left_eye[0])**2 + 
            (right_eye[1] - left_eye[1])**2
        )
        
        # Skip alignment if eyes are too close (likely detection error)
        if eye_distance < FaceRecognitionConfig.EYE_ALIGNMENT_THRESHOLD:
            return image.resize((face_size, face_size), Image.LANCZOS)
        
        # Calculate rotation angle
        angle = calculate_eye_angle(left_eye, right_eye)
        
        # Calculate face center (midpoint between eyes)
        face_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2
        )
        
        # Rotate image to align eyes horizontally
        aligned_image = image.rotate(
            -angle,
            center=face_center,
            expand=True,
            fillcolor=(128, 128, 128)
        )
        
        # Calculate desired eye positions in aligned image
        desired_left_eye = (face_size * 0.3, face_size * 0.4)
        desired_right_eye = (face_size * 0.7, face_size * 0.4)
        desired_eye_distance = desired_right_eye[0] - desired_left_eye[0]
        
        # Calculate scale factor
        scale = desired_eye_distance / eye_distance
        
        # Calculate crop/resize parameters
        rotated_width, rotated_height = aligned_image.size
        new_face_center = (rotated_width / 2, rotated_height / 2)
        crop_size = int(face_size / scale)
        
        # Center crop around face
        left = max(0, int(new_face_center[0] - crop_size / 2))
        top = max(0, int(new_face_center[1] - crop_size / 2))
        right = min(rotated_width, left + crop_size)
        bottom = min(rotated_height, top + crop_size)
        
        # Crop and resize to final size
        cropped_face = aligned_image.crop((left, top, right, bottom))
        final_face = cropped_face.resize((face_size, face_size), Image.LANCZOS)
        
        return final_face
        
    except Exception as e:
        logger.warning(f"Face alignment failed, using simple resize: {e}")
        # Fallback: simple resize without alignment
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image.resize((face_size, face_size), Image.LANCZOS)


def preprocess_face_for_facenet(face_image: Image.Image) -> torch.Tensor:
    """
    Preprocess face image for FaceNet model
    
    Args:
        face_image: PIL Image (112x112)
    
    Returns:
        Normalized tensor ready for FaceNet
    """
    try:
        # Convert to tensor and normalize to [-1, 1] range (FaceNet standard)
        face_tensor = torch.tensor(np.array(face_image)).permute(2, 0, 1).float()
        face_tensor = (face_tensor - 127.5) / 128.0  # Normalize to [-1, 1]
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        return face_tensor.to(DEVICE)
    except Exception as e:
        logger.error(f"Face preprocessing error: {e}")
        return None


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two face embeddings
    
    Args:
        embedding1: First face embedding
        embedding2: Second face embedding
    
    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    try:
        # Ensure embeddings are 2D arrays
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0, 0]
        
        # Convert to 0-1 range (cosine similarity is -1 to 1)
        similarity = (similarity + 1) / 2
        
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        
    except Exception as e:
        logger.error(f"Cosine similarity calculation error: {e}")
        return 0.0


def is_face_sufficiently_visible(
    face_image: np.ndarray,
    visibility_threshold: float = FaceRecognitionConfig.FACE_VISIBILITY_THRESHOLD,
    min_area: int = FaceRecognitionConfig.FACE_AREA_THRESHOLD
) -> bool:
    """
    Check if face is sufficiently visible and has good quality
    
    Args:
        face_image: OpenCV image (BGR)
        visibility_threshold: Minimum visibility ratio (0.0-1.0)
        min_area: Minimum face area in pixels
    
    Returns:
        True if face is good enough for recognition
    """
    try:
        height, width = face_image.shape[:2]
        face_area = height * width
        
        # Check minimum face area
        if face_area < min_area:
            return False
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Check for sufficient contrast/detail using standard deviation
        std_dev = np.std(gray)
        if std_dev < 15:  # Too uniform/blurry
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Face visibility check error: {e}")
        return False