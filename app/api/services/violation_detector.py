import cv2
import time
import logging
from datetime import datetime

from app.core.config import settings
from app.core.database import save_violation_to_database
from app.services.detection_service import trigger_hooter_with_duration

def process_frame_for_violations(detector, frame):
    """
    Contains the full, original detection logic for processing a single frame.
    This function is called by the OptimizedHairnetDetector class.
    
    Args:
        detector: The instance of the OptimizedHairnetDetector class.
        frame: The raw video frame to process.

    Returns:
        A tuple containing:
        - frame_annotated: The frame with bounding boxes and text drawn on it.
        - violations: A list of violation dictionaries.
        - valid_violation: A boolean indicating if a new violation was confirmed.
    """
    
    # 2. Prepare detection frame and zone area
    detection_frame = cv2.resize(frame, (settings.DETECTION_WIDTH, settings.DETECTION_HEIGHT))
    zone_frame, crop_y_offset = detector.crop_to_zone(detection_frame)
    
    if zone_frame.size == 0:
        return detector.draw_zone_lines(frame), [], False

    frame_annotated = detector.draw_zone_lines(frame.copy())
    scale_x = frame.shape[1] / settings.DETECTION_WIDTH
    scale_y = frame.shape[0] / settings.DETECTION_HEIGHT

    violations = []
    no_hairnet_detected = False
    highest_confidence = 0.0

    hairnet_boxes = []
    no_hairnet_boxes = []

    # 3. DETECT BOTH HAIRNET AND NO_HAIRNET
    hairnet_results = detector.hairnet_model(zone_frame, verbose=False)[0]
    if len(hairnet_results.boxes.data):
        for box in hairnet_results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            if conf < settings.CONFIDENCE_THRESHOLD:
                continue

            class_name = detector.hairnet_class_names[int(cls)]
            orig_x1 = int(x1 * scale_x)
            orig_y1 = int((y1 + crop_y_offset) * scale_y)
            orig_x2 = int(x2 * scale_x)
            orig_y2 = int((y2 + crop_y_offset) * scale_y)
            center_y = (orig_y1 + orig_y2) // 2
            
            if 200 <= center_y <= 550: # Using hardcoded values from original logic
                if class_name.lower() == "no_hairnet":
                    no_hairnet_boxes.append(box)
                    cv2.rectangle(frame_annotated, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 0, 255), 3)
                    cv2.putText(frame_annotated, f"NO HAIRNET! {conf:.2f}", (orig_x1, orig_y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    no_hairnet_detected = True
                    if conf > highest_confidence:
                        highest_confidence = conf
                elif class_name.lower() == "hairnet":
                    hairnet_boxes.append(box)
                    cv2.rectangle(frame_annotated, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
                    cv2.putText(frame_annotated, f"Hairnet {conf:.2f}", (orig_x1, orig_y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 4. IF NO HAIRNET DETECTED, DETECT FACES AND MAP
    faces_associated_with_no_hairnet = []
    if no_hairnet_detected:
        face_results = detector.face_model(zone_frame, verbose=False)[0]
        if len(face_results.boxes.data):
            for face_box in face_results.boxes.data.cpu().numpy():
                if face_box[4] < 0.5:
                    continue
                
                face_center_y = ((face_box[1] + face_box[3]) / 2.0 + crop_y_offset) * scale_y
                if 200 <= face_center_y <= 550:
                    for no_hairnet_box in no_hairnet_boxes:
                        overlap_x1 = max(face_box[0], no_hairnet_box[0])
                        overlap_y1 = max(face_box[1], no_hairnet_box[1])
                        overlap_x2 = min(face_box[2], no_hairnet_box[2])
                        overlap_y2 = min(face_box[3], no_hairnet_box[3])
                        
                        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                            face_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
                            if face_area > 0 and overlap_area / face_area >= 0.2:
                                faces_associated_with_no_hairnet.append(face_box)
                                break

    # 5. DETERMINE VIOLATION STATUS
    valid_violation = no_hairnet_detected and len(faces_associated_with_no_hairnet) > 0

    if valid_violation:
        if detector.no_hairnet_start_time is None:
            detector.no_hairnet_start_time = time.time()
        
        elif (time.time() - detector.no_hairnet_start_time >= settings.VIOLATION_TIME_SEC) and not detector.violation_active:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            violation_faces_data = detector.save_violation_faces(zone_frame, faces_associated_with_no_hairnet, timestamp, crop_y_offset)

            if violation_faces_data:
                img_path = settings.OUTPUT_DIR / f"violation_{timestamp}.jpg"
                cv2.imwrite(str(img_path), frame_annotated)
                
                video_path = detector.start_violation_video_recording(timestamp)
                if video_path:
                    detector.violation_active = True
                    detector.violation_count += 1
                    trigger_hooter_with_duration(settings.HOOTER_DURATION)

                    best_face = max(violation_faces_data, key=lambda x: x['face_confidence'])
                    log_data = {
                        "timestamp": timestamp, "employee_id": best_face.get('employee_id'), 
                        "employee_name": best_face.get('employee_name'), "confidence": highest_confidence, 
                        "face_confidence": best_face.get('face_confidence'), "violation_image": str(img_path), 
                        "violation_faces": ";".join([d['file_path'] for d in violation_faces_data]), 
                        "violation_video": video_path, 
                        "original_known_faces": "" 
                    }
                    save_violation_to_database(**log_data)
                    logging.info(f"VIOLATION {detector.violation_count} LOGGED for {best_face.get('employee_name') or 'Unknown'}")
    else:
        if detector.violation_active: detector.reset_violation_state()
        detector.no_hairnet_start_time = None

    if detector.recording_violation_video:
        detector.write_violation_frame(frame_annotated)
        
    return frame_annotated, violations, valid_violation