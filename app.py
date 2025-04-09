import cv2
import traceback
from deepface import DeepFace
import time
import numpy as np
import threading
import queue
emotion_colors = {
    'angry': (0, 0, 255),
    'disgust': (0, 128, 0),
    'fear': (139, 0, 139),
    'happy': (0, 255, 0),
    'sad': (255, 0, 0),
    'surprise': (0, 255, 255),
    'neutral': (255, 255, 255),
    'No Face': (128, 128, 128),
    'Error': (0, 0, 0),
    'Unknown': (128, 128, 128)
}
default_color = (128, 128, 128)
detector_backend = 'mtcnn'
analysis_interval = 0.25

frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

def analysis_worker():
    print("[ANALYSIS THREAD] Started.")
    last_analysis_time = 0
    
    while not stop_event.is_set():
        try:
            raw_frame = frame_queue.get_nowait() 
        except queue.Empty:
            time.sleep(0.01)
            continue

        current_time = time.time()
        if current_time - last_analysis_time < analysis_interval:
            try:
                if result_queue.full(): result_queue.get_nowait()
                result_queue.put_nowait(raw_frame) 
            except queue.Full: pass
            time.sleep(0.01) 
            continue

        last_analysis_time = current_time
        annotated_frame = raw_frame.copy()

        try:
            results = DeepFace.analyze(
                img_path=raw_frame,
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
                        try:
                            x, y, w, h = region['x'], region['y'], region['w'], region['h'] 
                            confidence = emotions.get(dominant_emotion, 0)

                            color = emotion_colors.get(dominant_emotion, default_color) 

                            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)

                            text = f"{dominant_emotion} ({confidence:.2f})"

                            text_y = y - 10 if y - 10 > 10 else y + 10
                            cv2.putText(annotated_frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        except KeyError as ke:
                             print(f"[ANALYSIS THREAD] Warning: Missing key in region dict: {ke}")
                        except Exception as draw_e:
                             print(f"[ANALYSIS THREAD] Error during drawing: {draw_e}")
                
            if result_queue.full(): result_queue.get_nowait()
            result_queue.put_nowait(annotated_frame)

        except Exception as e:
            print(f"[ANALYSIS THREAD] Error during analysis loop:")
            print(f"  Error Type: {type(e).__name__}")
            print(f"  Error Details: {e}")
            print("--- Traceback ---")
            traceback.print_exc() 
            print("-----------------")

            try:
                if result_queue.full(): result_queue.get_nowait()
                result_queue.put_nowait(raw_frame) 
            except queue.Full: pass
            time.sleep(0.5)


if __name__ == "__main__":
    
    worker_thread = threading.Thread(target=analysis_worker, daemon=True)
    worker_thread.start()

    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        stop_event.set()
        exit()

    print("Starting real-time analysis... Press 'q' to quit.")
    prev_time = time.time()
    latest_processed_frame = None

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            stop_event.set()
            break

        try:
            if frame_queue.full():
                frame_queue.get_nowait() 
            frame_queue.put_nowait(frame) 
        except queue.Full:
            pass 

        try:
            latest_processed_frame = result_queue.get_nowait() 
        except queue.Empty:
            display_frame = latest_processed_frame if latest_processed_frame is not None else frame

        if latest_processed_frame is not None:
             display_frame = latest_processed_frame

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Real-time Emotion Analysis (Press Q to Quit)', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            stop_event.set()
            break

    worker_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    print("Analysis stopped.")
