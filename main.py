from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import tempfile
import os
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

app = FastAPI(title="Video Emotion Detector with Confidence Scoring")

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def calculate_emotion_confidence(landmarks) -> Tuple[str, float]:
    """Calculate emotion and confidence score (0-1)"""
    # Mouth features
    mouth_open = landmarks[14].y - landmarks[13].y
    mouth_width = landmarks[308].x - landmarks[78].x
    
    # Eye features
    left_eye = landmarks[145].y - landmarks[159].y
    right_eye = landmarks[374].y - landmarks[386].y
    eye_open = (left_eye + right_eye) / 2
    
    # Eyebrow features
    eyebrow_left = landmarks[336].y
    eyebrow_right = landmarks[66].y
    eyebrow_avg = (eyebrow_left + eyebrow_right) / 2
    
    # Calculate emotion probabilities
    emotions = {
        'happy': min(1.0, mouth_open * 10 + mouth_width * 2),
        'angry': min(1.0, (0.3 - eyebrow_avg) * 3 + (0.03 - eye_open) * 20),
        'sad': min(1.0, (0.02 - mouth_open) * 30 + (0.03 - eye_open) * 20),
        'surprised': min(1.0, eye_open * 10 + mouth_open * 5),
        'neutral': 0.5  # Baseline
    }
    
    # Normalize probabilities
    total = sum(emotions.values())
    normalized = {k: v/total for k, v in emotions.items()}
    
    # Get dominant emotion
    dominant = max(normalized.items(), key=lambda x: x[1])
    return dominant[0], round(dominant[1], 2)

@app.post("/detect-emotions/")
async def detect_emotions(file: UploadFile = File(...), frame_interval: int = 5):
    """Process video with confidence scoring"""
    temp_path = None
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            await file.seek(0)
            tmp.write(await file.read())
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(400, detail="Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = 0
        results = []
        emotion_stats = defaultdict(list)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = face_mesh.process(rgb_frame)

                if face_results.multi_face_landmarks:
                    emotion, confidence = calculate_emotion_confidence(
                        face_results.multi_face_landmarks[0].landmark
                    )
                    
                    results.append({
                        "frame": frame_count,
                        "time": round(frame_count/fps, 2),
                        "emotion": emotion,
                        "confidence": confidence
                    })
                    emotion_stats[emotion].append(confidence)

            except Exception as e:
                print(f"Frame {frame_count} error: {str(e)}")
                continue

        cap.release()

        if not results:
            raise HTTPException(400, detail="No faces detected")

        # Calculate aggregate confidence per emotion
        emotion_summary = {}
        for emotion, confidences in emotion_stats.items():
            avg_conf = round(np.mean(confidences), 2)
            emotion_summary[emotion] = {
                "percentage": round(len(confidences)/len(results)*100, 1),
                "avg_confidence": avg_conf
            }

        return {
            "status": "success",
            "total_frames": len(results),
            "emotion_summary": emotion_summary,
            "frame_samples": results[:5]  # Sample frames
        }

    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)