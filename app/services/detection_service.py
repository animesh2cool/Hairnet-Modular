"""
Optimized Hairnet Detection Service
Core detection engine with face recognition and video recording
"""
import cv2
import csv
import time
import pickle
import logging
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import deque

from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import requests

from ..core.config import (
    DetectionConfig, FaceRecognitionConfig, VideoRecordingConfig,
    VideoConfig, DirectoryConfig, ModelConfig, HooterConfig,
    DEVICE, hooter_config
)
from ..core.database import save_violation_to_database
from .face_utils import (
    align_face_by_eyes, preprocess_face_for_facenet,
    calculate_cosine_similarity, is_face_sufficiently_visible
)

logger = logging.getLogger(__name__)


class OptimizedHairnetDetector:
    """
    Optimized Hairnet Detection with Face Recognition
    """
    
    def __init__(self, output_dir: Path, violation_faces_dir: Path, violation_video_dir: Path):
        """Initialize detector with all configurations"""
        
        # Directories
        self.output_dir = output_dir
        self.violation_faces_dir = violation_faces_dir
        self.violation_video_dir = violation_video_dir
        
        # Load models
        self.hairnet_model = YOLO(ModelConfig.HAIRNET_MODEL_PATH)
        self.face_model = YOLO(ModelConfig.FACE_MODEL_PATH)
        self.hairnet_class_names = self.hairnet_model.names
        
        # Face recognition setup
        self.setup_face_recognition()
        
        # Violation tracking
        self.no_hairnet_start_time = None
        self.violation_active = False
        self.violation_count = 0
        self.violation_face_save_count = 0
        self.current_violation_saved_faces = set()
        
        # Video recording
        self.recording_violation_video = False
        self.violation_video_writer = None
        self.violation_video_path = None
        self.violation_start_time = None
        self.frames_recorded_post_violation = 0
        self.video_codec = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Hooter state
        self.hooter_active = False
        self.hooter_last_triggered = 0
        self.hooter_auto_off_timer = None
        
        # Performance optimization
        self.last_detection_time = 0
        self.detection_interval = 1.0 / VideoConfig.PROCESSING_FPS
        
        # Zone boundaries
        self.zone_y_min = max(0, DetectionConfig.LINE_Y_MIN - DetectionConfig.ZONE_MARGIN)
        self.zone_y_max = min(
            VideoConfig.DETECTION_HEIGHT,
            DetectionConfig.LINE_Y_MAX + DetectionConfig.ZONE_MARGIN
        )
        
        # Face visibility parameters
        self.min_face_visibility = FaceRecognitionConfig.FACE_VISIBILITY_THRESHOLD
        self.face_area_threshold = FaceRecognitionConfig.FACE_AREA_THRESHOLD
        
        # Face tracking (DISABLED)
        self.face_tracks = {}
        self.entry_count = 0
        self.track_faces_enabled = False
        
        # CSV log setup
        self.log_file = output_dir / "violation_log.csv"
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Employee_ID', 'Employee_Name', 'Confidence',
                'Face_Confidence', 'Violation_Image', 'Violation_Faces',
                'Violation_Video', 'Original_Known_Faces'
            ])
    
    def setup_face_recognition(self):
        """Initialize face recognition with MTCNN and FaceNet"""
        try:
            logger.info("🔍 Initializing face recognition with alignment...")
            
            # Initialize MTCNN
            self.mtcnn = MTCNN(
                image_size=FaceRecognitionConfig.FACE_ALIGN_SIZE,
                margin=20,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=DEVICE,
                keep_all=False,
                select_largest=True
            )
            
            # Initialize FaceNet
            self.resnet = InceptionResnetV1(
                pretrained=ModelConfig.FACENET_PRETRAINED,
                classify=False,
                device=DEVICE
            ).eval()
            
            # Load known faces
            self.load_known_faces_with_alignment()
            
            logger.info(f"✅ Face recognition initialized with {len(self.known_embeddings)} known faces")
            
        except Exception as e:
            logger.error(f"❌ Face recognition initialization failed: {e}")
            self.mtcnn = None
            self.resnet = None
            self.known_embeddings = {}
            self.known_names = {}
    
    def load_known_faces_with_alignment(self):
        """Load known face embeddings with alignment"""
        self.known_embeddings = {}
        self.known_names = {}
        
        embeddings_file = DirectoryConfig.KNOWN_FACES_DIR / f"aligned_face_embeddings_{FaceRecognitionConfig.FACE_INPUT_SIZE}.pkl"
        cached_data = {}
        
        # Load cache
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"📁 Found cached embeddings: {len(cached_data.get('embeddings', {}))}")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Scan for changes
        current_folders = set()
        current_images = {}
        
        for person_folder in DirectoryConfig.KNOWN_FACES_DIR.iterdir():
            if not person_folder.is_dir():
                continue
            
            folder_name = person_folder.name
            current_folders.add(folder_name)
            
            image_files = set()
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
                image_files.update([f.name for f in person_folder.glob(ext)])
            
            current_images[folder_name] = image_files
        
        # Check if update needed
        cached_folders = set(cached_data.get('folder_structure', {}).keys())
        needs_update = bool(current_folders - cached_folders) or not cached_data
        
        if not needs_update:
            for folder in current_folders.intersection(cached_folders):
                if current_images[folder] != set(cached_data.get('folder_structure', {}).get(folder, [])):
                    needs_update = True
                    break
        
        # Use cache or process
        if not needs_update and cached_data:
            self.known_embeddings = cached_data.get('embeddings', {})
            self.known_names = cached_data.get('names', {})
            logger.info("✅ Using cached embeddings")
            return
        
        logger.info("🔄 Processing faces with alignment...")
        
        # Start with cache
        if cached_data:
            self.known_embeddings = cached_data.get('embeddings', {})
            self.known_names = cached_data.get('names', {})
        
        # Process changed folders
        folders_to_process = list(current_folders - cached_folders) if cached_data else list(current_folders)
        
        for folder_name in folders_to_process:
            self._process_person_folder(folder_name, current_images)
        
        # Update cache
        if self.known_embeddings:
            self._save_embeddings_cache(embeddings_file, current_images)
        
        logger.info(f"✅ Face loading complete: {len(self.known_embeddings)} employees")
    
    def _process_person_folder(self, folder_name: str, current_images: dict):
        """Process single person folder for face embeddings"""
        person_folder = DirectoryConfig.KNOWN_FACES_DIR / folder_name
        if not person_folder.exists():
            return
        
        # Parse employee info
        try:
            parts = folder_name.split('_', 1)
            employee_id = parts[0] if len(parts) == 2 else folder_name
            employee_name = parts[1].replace('_', ' ') if len(parts) == 2 else folder_name
        except:
            employee_id = folder_name
            employee_name = folder_name
        
        # Process images
        person_embeddings = []
        image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
            image_files.extend(list(person_folder.glob(ext)))
        
        successful = 0
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
                
                if boxes is not None and len(boxes) > 0 and landmarks is not None:
                    if probs[0] > 0.8:
                        left_eye = (landmarks[0][0][0], landmarks[0][0][1])
                        right_eye = (landmarks[0][1][0], landmarks[0][1][1])
                        
                        # Crop with margin
                        x1, y1, x2, y2 = boxes[0].astype(int)
                        margin = 20
                        face_img = img.crop((
                            max(0, x1-margin),
                            max(0, y1-margin),
                            min(img.width, x2+margin),
                            min(img.height, y2+margin)
                        ))
                        
                        # Adjust eye coordinates
                        crop_left_eye = (left_eye[0] - max(0, x1-margin), left_eye[1] - max(0, y1-margin))
                        crop_right_eye = (right_eye[0] - max(0, x1-margin), right_eye[1] - max(0, y1-margin))
                        
                        # Align and preprocess
                        aligned_face = align_face_by_eyes(face_img, crop_left_eye, crop_right_eye)
                        face_tensor = preprocess_face_for_facenet(aligned_face)
                        
                        if face_tensor is not None:
                            with torch.no_grad():
                                embedding = self.resnet(face_tensor).detach().cpu().numpy().flatten()
                                embedding = embedding / np.linalg.norm(embedding)
                                person_embeddings.append(embedding)
                                successful += 1
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
        
        if person_embeddings:
            avg_embedding = np.mean(person_embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            self.known_embeddings[employee_id] = avg_embedding
            self.known_names[employee_id] = employee_name
            logger.info(f"✅ {employee_name} ({employee_id}) - {successful}/{len(image_files)} faces")
    
    def _save_embeddings_cache(self, file_path: Path, current_images: dict):
        """Save embeddings to cache file"""
        try:
            cache_data = {
                'embeddings': self.known_embeddings,
                'names': self.known_names,
                'folder_structure': current_images,
                'last_updated': datetime.now().isoformat(),
                'alignment_enabled': True,
                'face_size': FaceRecognitionConfig.FACE_INPUT_SIZE,
                'model': 'FaceNet_aligned'
            }
            with open(file_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"💾 Cache updated with {len(self.known_embeddings)} employees")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def recognize_face_with_alignment(self, face_image: np.ndarray) -> Tuple[Optional[str], str, float]:
        """
        Recognize face with alignment
        Returns: (employee_id, employee_name, confidence)
        """
        if not self.mtcnn or not self.resnet or not self.known_embeddings:
            return None, "Unknown", 0.0
        
        try:
            if not is_face_sufficiently_visible(face_image):
                return None, "Unknown", 0.0
            
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
            
            boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
            
            if boxes is None or len(boxes) == 0 or landmarks is None:
                return None, "Unknown", 0.0
            
            if probs[0] < FaceRecognitionConfig.FACE_DETECTION_CONFIDENCE:
                return None, "Unknown", 0.0
            
            left_eye = (landmarks[0][0][0], landmarks[0][0][1])
            right_eye = (landmarks[0][1][0], landmarks[0][1][1])
            
            aligned_face = align_face_by_eyes(pil_image, left_eye, right_eye)
            face_tensor = preprocess_face_for_facenet(aligned_face)
            
            if face_tensor is None:
                return None, "Unknown", 0.0
            
            with torch.no_grad():
                embedding = self.resnet(face_tensor).detach().cpu().numpy().flatten()
                embedding = embedding / np.linalg.norm(embedding)
            
            best_match_id = None
            best_match_name = "Unknown"
            best_similarity = 0.0
            
            for employee_id, known_embedding in self.known_embeddings.items():
                similarity = calculate_cosine_similarity(embedding, known_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = employee_id
                    best_match_name = self.known_names.get(employee_id, employee_id)
            
            if best_similarity >= FaceRecognitionConfig.FACE_RECOGNITION_THRESHOLD:
                return best_match_id, best_match_name, best_similarity
            else:
                return None, "Unknown", best_similarity
                
        except Exception as e:
            logger.error(f"Face recognition error: {e}")
            return None, "Unknown", 0.0

    def is_in_zone(self, center_y: float, margin: int = 0) -> bool:
        """Check if detection is within compliance zone"""
        return (DetectionConfig.LINE_Y_MIN - margin) <= center_y <= (DetectionConfig.LINE_Y_MAX + margin)
    
    def crop_to_zone(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Crop frame to compliance zone"""
        height, width = frame.shape[:2]
        
        crop_y_start = max(0, int((DetectionConfig.LINE_Y_MIN - DetectionConfig.ZONE_MARGIN) * height / VideoConfig.STREAM_HEIGHT))
        crop_y_end = min(height, int((DetectionConfig.LINE_Y_MAX + DetectionConfig.ZONE_MARGIN) * height / VideoConfig.STREAM_HEIGHT))
        
        if crop_y_end <= crop_y_start:
            crop_y_start = 0
            crop_y_end = height
        
        cropped = frame[crop_y_start:crop_y_end, :]
        return cropped, crop_y_start
    
    def draw_zone_lines(self, frame: np.ndarray) -> np.ndarray:
        """Draw compliance zone lines on frame"""
        frame_height, frame_width = frame.shape[:2]
        
        zone_y_min = int(DetectionConfig.LINE_Y_MIN * frame_height / VideoConfig.STREAM_HEIGHT)
        zone_y_max = int(DetectionConfig.LINE_Y_MAX * frame_height / VideoConfig.STREAM_HEIGHT)
        
        zone_y_min = max(0, min(frame_height - 1, zone_y_min))
        zone_y_max = max(0, min(frame_height - 1, zone_y_max))
        
        cv2.line(frame, (0, zone_y_min), (frame_width, zone_y_min), (255, 0, 0), 4)
        cv2.line(frame, (0, zone_y_max), (frame_width, zone_y_max), (255, 0, 0), 4)
        
        cv2.putText(frame, "NON-COMPLIANCE ZONE", (10, zone_y_max + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        
        return frame
    
    def start_violation_video_recording(self, timestamp: str, video_buffer: list) -> Optional[str]:
        """Start recording violation video with buffered frames"""
        if self.recording_violation_video:
            return None
        
        try:
            self.violation_video_path = self.violation_video_dir / f"violation_video_{timestamp}.mp4"
            
            self.violation_video_writer = cv2.VideoWriter(
                str(self.violation_video_path),
                self.video_codec,
                VideoRecordingConfig.VIDEO_BUFFER_FPS,
                (VideoConfig.STREAM_WIDTH, VideoConfig.STREAM_HEIGHT)
            )
            
            if not self.violation_video_writer.isOpened():
                logger.error("Failed to open video writer")
                return None
            
            # Write buffered frames
            for buffered_frame in video_buffer:
                self.violation_video_writer.write(buffered_frame)
            
            self.recording_violation_video = True
            self.violation_start_time = time.time()
            self.frames_recorded_post_violation = 0
            
            logger.info(f"Started video recording: {self.violation_video_path}")
            return str(self.violation_video_path)
            
        except Exception as e:
            logger.exception(f"Error starting video: {e}")
            return None
    
    def write_violation_frame(self, frame: np.ndarray):
        """Write frame to violation video"""
        if not self.recording_violation_video or not self.violation_video_writer:
            return
        
        try:
            self.violation_video_writer.write(frame)
            self.frames_recorded_post_violation += 1
            
            if self.frames_recorded_post_violation >= (VideoRecordingConfig.VIDEO_POST_SECONDS * VideoRecordingConfig.VIDEO_BUFFER_FPS):
                self.stop_violation_video_recording()
                
        except Exception as e:
            logger.exception(f"Error writing frame: {e}")
            self.stop_violation_video_recording()
    
    def stop_violation_video_recording(self):
        """Stop violation video recording"""
        if not self.recording_violation_video:
            return
        
        try:
            if self.violation_video_writer:
                self.violation_video_writer.release()
                self.violation_video_writer = None
            
            self.recording_violation_video = False
            duration = time.time() - self.violation_start_time if self.violation_start_time else 0
            logger.info(f"Stopped video recording. Duration: {duration:.2f}s")
            
        except Exception as e:
            logger.exception(f"Error stopping video: {e}")
    
    def save_violation_faces(self, frame: np.ndarray, face_boxes: list, 
                            violation_timestamp: str, crop_offset: int = 0) -> list:
        """Save faces during violations with visibility check"""
        self.current_violation_saved_faces.clear()
        violation_faces_data = []
        
        for i, box in enumerate(face_boxes):
            x1, y1, x2, y2, conf, cls = box
            if conf < 0.5:
                continue
            
            center_y_det = (y1 + y2) / 2.0 + crop_offset
            center_y_stream = center_y_det * (VideoConfig.STREAM_HEIGHT / VideoConfig.DETECTION_HEIGHT)
            
            if not self.is_in_zone(center_y_stream):
                continue
            
            face_y1 = max(0, int(y1))
            face_y2 = min(frame.shape[0], int(y2))
            face_x1 = max(0, int(x1))
            face_x2 = min(frame.shape[1], int(x2))
            
            if face_y2 <= face_y1 or face_x2 <= face_x1:
                continue
            
            face_region = frame[face_y1:face_y2, face_x1:face_x2]
            if face_region.size > 0:
                try:
                    if not is_face_sufficiently_visible(face_region, self.min_face_visibility):
                        logger.info("[FACE SKIP] Visibility <70%")
                        continue
                    
                    employee_id, employee_name, face_confidence = self.recognize_face_with_alignment(face_region)
                    
                    if employee_id and employee_id in self.current_violation_saved_faces:
                        logger.info(f"[FACE SKIP] {employee_name} already saved")
                        continue
                    
                    if not employee_id:
                        unknown_count = sum(1 for f in violation_faces_data if not f['employee_id'])
                        if unknown_count >= 1:
                            logger.info("[FACE SKIP] Unknown face limit reached")
                            continue
                    
                    if employee_id:
                        face_filename = self.violation_faces_dir / f"violation_{violation_timestamp}_{employee_id}_{employee_name.replace(' ', '_')}_{self.violation_face_save_count:04d}.jpg"
                    else:
                        face_filename = self.violation_faces_dir / f"violation_{violation_timestamp}_UNKNOWN_{self.violation_face_save_count:04d}.jpg"
                    
                    success = cv2.imwrite(str(face_filename), face_region)
                    
                    if success:
                        if employee_id:
                            self.current_violation_saved_faces.add(employee_id)
                        
                        violation_faces_data.append({
                            'file_path': str(face_filename),
                            'employee_id': employee_id,
                            'employee_name': employee_name,
                            'face_confidence': face_confidence
                        })
                        
                        self.violation_face_save_count += 1
                        logger.info(f"[FACE SAVED] {employee_name} ({employee_id}) - Conf: {face_confidence:.3f}")
                        
                except Exception as e:
                    logger.error(f"[FACE ERROR] {e}")
        
        return violation_faces_data
    
    def get_original_known_faces_for_violation(self, violation_faces_data: list) -> str:
        """Get original known face paths for violation"""
        original_paths = []
        
        for face_data in violation_faces_data:
            employee_id = face_data.get('employee_id')
            
            if employee_id and employee_id in self.known_names:
                employee_name = self.known_names[employee_id]
                folder_name = f"{employee_id}_{employee_name.replace(' ', '_')}"
                person_folder = DirectoryConfig.KNOWN_FACES_DIR / folder_name
                
                if person_folder.exists():
                    images = list(person_folder.glob("*.jpg")) + list(person_folder.glob("*.png"))
                    if images:
                        original_paths.append(str(images[0]))
        
        unique_paths = []
        for path in original_paths:
            if path not in unique_paths:
                unique_paths.append(path)
        
        return ';'.join(unique_paths) if unique_paths else ''
    
    def reset_violation_state(self):
        """Reset violation state"""
        self.current_violation_saved_faces.clear()
        self.violation_active = False
        self.no_hairnet_start_time = None
    
    def trigger_hooter(self, duration: int = None):
        """
        Trigger hooter for specified duration with master switch check
        Will NOT trigger if HOOTER_MASTER_SWITCH is set to 'OFF' in .env
        """
        # MASTER SWITCH CHECK - Hard stop if disabled
        if not hooter_config.is_enabled():
            logger.info(f"⚠️ Hooter trigger blocked - Master switch is OFF")
            return False
        
        # Check cooldown period
        current_time = time.time()
        time_since_last_trigger = current_time - self.hooter_last_triggered
        
        if self.hooter_active or time_since_last_trigger < hooter_config.COOLDOWN_SECONDS:
            remaining = int(hooter_config.COOLDOWN_SECONDS - time_since_last_trigger)
            logger.info(f"⏳ Hooter on cooldown ({remaining}s remaining)")
            return False
        
        duration = duration or hooter_config.DURATION
        
        try:
            response = requests.get(hooter_config.ON_URL, timeout=hooter_config.TIMEOUT)
            response.raise_for_status()
            
            self.hooter_active = True
            self.hooter_last_triggered = current_time
            
            logger.info(f"🚨 Hooter ON for {duration}s (cooldown: {hooter_config.COOLDOWN_SECONDS}s)")
            
            def turn_off():
                self.turn_off_hooter()
                self.hooter_active = False
            
            self.hooter_auto_off_timer = threading.Timer(duration, turn_off)
            self.hooter_auto_off_timer.daemon = True
            self.hooter_auto_off_timer.start()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Hooter activation failed: {e}")
            return False
    
    def turn_off_hooter(self):
        """
        Turn off hooter
        Always works regardless of master switch (safety feature)
        """
        try:
            response = requests.get(hooter_config.OFF_URL, timeout=hooter_config.TIMEOUT)
            response.raise_for_status()
            logger.info("✅ Hooter OFF")
            return True
        except Exception as e:
            logger.error(f"❌ Hooter deactivation failed: {e}")
            return False
    
    def update_face_tracks_and_count(self, frame: np.ndarray) -> list:
        """DISABLED - No entry tracking"""
        return []


    def detect_zone_only(self, frame: np.ndarray, video_buffer: list) -> Tuple[np.ndarray, list, bool]:
        """
        Main detection logic with zone focus
        Returns: (annotated_frame, violations, zone_active)
        """
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_detection_time < self.detection_interval:
            frame_with_lines = self.draw_zone_lines(frame)
            if self.recording_violation_video:
                self.write_violation_frame(frame_with_lines)
            return frame_with_lines, [], False
        
        self.last_detection_time = current_time
        
        # Prepare detection
        detection_frame = cv2.resize(frame, (VideoConfig.DETECTION_WIDTH, VideoConfig.DETECTION_HEIGHT))
        zone_frame, crop_y_offset = self.crop_to_zone(detection_frame)
        
        if zone_frame.size == 0:
            return self.draw_zone_lines(frame), [], False
        
        frame_annotated = self.draw_zone_lines(frame.copy())
        frame_height, frame_width = frame.shape[:2]
        scale_x = frame_width / VideoConfig.DETECTION_WIDTH
        scale_y = frame_height / VideoConfig.DETECTION_HEIGHT
        
        violations = []
        no_hairnet_detected = False
        highest_confidence = 0.0
        violation_center_y = 0
        
        hairnet_boxes = []
        no_hairnet_boxes = []
        
        # Detect hairnet/no_hairnet
        hairnet_results = self.hairnet_model(zone_frame, verbose=False)[0]
        if len(hairnet_results.boxes.data):
            for box in hairnet_results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box
                conf = float(conf)
                
                if conf < DetectionConfig.CONFIDENCE_THRESHOLD:
                    continue
                
                class_name = self.hairnet_class_names[int(cls)]
                
                orig_x1 = int(x1 * scale_x)
                orig_y1 = int((y1 + crop_y_offset) * scale_y)
                orig_x2 = int(x2 * scale_x)
                orig_y2 = int((y2 + crop_y_offset) * scale_y)
                center_y = (orig_y1 + orig_y2) // 2
                stream_center_y = center_y * VideoConfig.STREAM_HEIGHT / frame_height
                
                if DetectionConfig.LINE_Y_MIN <= stream_center_y <= DetectionConfig.LINE_Y_MAX:
                    if class_name.lower() == "no_hairnet":
                        no_hairnet_boxes.append((x1, y1, x2, y2, conf))
                        cv2.rectangle(frame_annotated, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 0, 255), 3)
                        cv2.putText(frame_annotated, f"NO HAIRNET! {conf:.2f}",
                                   (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        no_hairnet_detected = True
                        
                        if conf > highest_confidence:
                            highest_confidence = conf
                            violation_center_y = center_y
                        
                        violations.append({
                            "type": "no_hairnet",
                            "confidence": conf,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "center": [orig_x1, orig_y1],
                            "in_zone": True
                        })
                    elif class_name.lower() == "hairnet":
                        hairnet_boxes.append((x1, y1, x2, y2, conf))
                        cv2.rectangle(frame_annotated, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
                        cv2.putText(frame_annotated, f"Hairnet {conf:.2f}",
                                   (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Detect faces if no_hairnet found
        faces_associated_with_no_hairnet = []
        if no_hairnet_detected:
            face_results = self.face_model(zone_frame, verbose=False)[0]
            if len(face_results.boxes.data):
                for face_box in face_results.boxes.data.cpu().numpy():
                    face_x1, face_y1, face_x2, face_y2, face_conf, face_cls = face_box
                    face_conf = float(face_conf)
                    
                    if face_conf < 0.5:
                        continue
                    
                    face_center_y = (face_y1 + face_y2) / 2.0 + crop_y_offset
                    stream_center_y_face = face_center_y * (VideoConfig.STREAM_HEIGHT / VideoConfig.DETECTION_HEIGHT)
                    
                    if DetectionConfig.LINE_Y_MIN <= stream_center_y_face <= DetectionConfig.LINE_Y_MAX:
                        # Check overlap with no_hairnet
                        for nh_x1, nh_y1, nh_x2, nh_y2, nh_conf in no_hairnet_boxes:
                            overlap_x1 = max(face_x1, nh_x1)
                            overlap_y1 = max(face_y1, nh_y1)
                            overlap_x2 = min(face_x2, nh_x2)
                            overlap_y2 = min(face_y2, nh_y2)
                            
                            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                                face_area = (face_x2 - face_x1) * (face_y2 - face_y1)
                                
                                if overlap_area / face_area >= 0.2:
                                    # Check visibility
                                    face_y1_int = max(0, int(face_y1))
                                    face_y2_int = min(zone_frame.shape[0], int(face_y2))
                                    face_x1_int = max(0, int(face_x1))
                                    face_x2_int = min(zone_frame.shape[1], int(face_x2))
                                    
                                    if face_y2_int > face_y1_int and face_x2_int > face_x1_int:
                                        face_region = zone_frame[face_y1_int:face_y2_int, face_x1_int:face_x2_int]
                                        if face_region.size > 0:
                                            if is_face_sufficiently_visible(face_region, self.min_face_visibility):
                                                faces_associated_with_no_hairnet.append([
                                                    face_x1, face_y1, face_x2, face_y2, face_conf, face_cls
                                                ])
                                                break
        
        # Determine violation status
        valid_violation = no_hairnet_detected and len(faces_associated_with_no_hairnet) > 0
        
        if valid_violation:
            cv2.putText(frame_annotated, f"VALID VIOLATION: {len(faces_associated_with_no_hairnet)} faces >70% visible",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Start violation timer
            if self.no_hairnet_start_time is None:
                self.no_hairnet_start_time = current_time
                
                # TRIGGER HOOTER - Only if master switch is ON
                if hooter_config.is_enabled() and not self.hooter_active:
                    self.trigger_hooter()
                elif not hooter_config.is_enabled():
                    logger.debug("Hooter not triggered - Master switch is OFF")
            
            elif (current_time - self.no_hairnet_start_time >= DetectionConfig.VIOLATION_TIME_SEC) and not self.violation_active:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save faces
                violation_faces_data = self.save_violation_faces(
                    zone_frame, faces_associated_with_no_hairnet, timestamp, crop_y_offset
                )
                
                if violation_faces_data:
                    img_path = self.output_dir / f"violation_{timestamp}.jpg"
                    cv2.imwrite(str(img_path), frame_annotated)
                    
                    # Start video recording
                    video_path = self.start_violation_video_recording(timestamp, video_buffer)
                    
                    if video_path:
                        if not self.hooter_active:
                            self.trigger_hooter(hooter_config.DURATION)
                        
                        best_face = max(violation_faces_data, key=lambda x: x['face_confidence'])
                        primary_employee_id = best_face.get('employee_id')
                        primary_employee_name = best_face.get('employee_name')
                        primary_face_confidence = float(best_face['face_confidence'])
                        
                        self.violation_count += 1
                        
                        # Prepare data
                        face_files = [f['file_path'] for f in violation_faces_data]
                        face_files_str = ';'.join(face_files)
                        original_known_faces_str = self.get_original_known_faces_for_violation(violation_faces_data)
                        
                        # CSV logging
                        with open(self.log_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                timestamp,
                                primary_employee_id or 'Unknown',
                                primary_employee_name or 'Unknown',
                                f"{highest_confidence:.2f}",
                                f"{primary_face_confidence:.3f}",
                                str(img_path),
                                face_files_str,
                                video_path,
                                original_known_faces_str
                            ])
                        
                        # Database logging
                        try:
                            save_violation_to_database(
                                timestamp=timestamp,
                                employee_id=primary_employee_id,
                                employee_name=primary_employee_name,
                                confidence=highest_confidence,
                                face_confidence=primary_face_confidence,
                                violation_image=str(img_path),
                                violation_faces=face_files_str,
                                violation_video=video_path,
                                original_known_faces=original_known_faces_str
                            )
                        except Exception as db_error:
                            logger.error(f"Database save failed: {db_error}")
                        
                        self.violation_active = True
                        logger.info(f"[VIOLATION] {primary_employee_name} ({primary_employee_id}) - Total: {self.violation_count}")
                    else:
                        logger.warning("[VIDEO FAILED] Not counting violation")
                else:
                    logger.info("[NO VALID FACES] Not counting violation")
        else:
            if no_hairnet_detected and not faces_associated_with_no_hairnet:
                cv2.putText(frame_annotated, "NO VALID FACES >70% VISIBLE",
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif not no_hairnet_detected:
                cv2.putText(frame_annotated, "ALL COMPLIANT",
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.hooter_active:
                self.turn_off_hooter()
                self.hooter_active = False
            
            self.no_hairnet_start_time = None
            if self.violation_active:
                self.reset_violation_state()
        
        # Continue video recording
        if self.recording_violation_video:
            self.write_violation_frame(frame_annotated)
        
        return frame_annotated, violations, valid_violation