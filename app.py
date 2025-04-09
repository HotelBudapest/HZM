import cv2
from deepface import DeepFace
import time
import numpy as np

detector_backend = 'opencv' 

skip_frames = 3 
frame_count = 0

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


print("Starting real-time analysis... Press 'q' to quit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    frame_count += 1
    
    if frame_count % skip_frames == 0:
        try:
            results = DeepFace.analyze(
                img_path=frame, 
                actions=['emotion'], 
                enforce_detection=False,
                detector_backend=detector_backend,
                silent=True
            )

            if results and isinstance(results, list) and len(results) > 0:
                for face_result in results:
                    region = face_result.get('region')
                    emotions = face_result.get('emotion')
                    dominant_emotion = face_result.get('dominant_emotion')

                    if region and emotions and dominant_emotion:
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        confidence = emotions.get(dominant_emotion, 0)
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        text = f"{dominant_emotion} ({confidence:.2f})"

                        text_y = y - 10 if y - 10 > 10 else y + 10
                        cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        

        except Exception as e:
            print(f"Error during DeepFace analysis: {e}") 

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Real-time Emotion Analysis (Press Q to Quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Analysis stopped.")
