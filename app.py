import cv2
import traceback
from deepface import DeepFace
import time
import numpy as np
import threading
import queue # Use queue for thread-safe communication
# --- Emotion to Color Mapping (BGR format for OpenCV) ---
emotion_colors = {
    'angry': (0, 0, 255),    # Red
    'disgust': (0, 128, 0),  # Dark Green (less common)
    'fear': (139, 0, 139),   # Dark Magenta
    'happy': (0, 255, 0),    # Lime Green
    'sad': (255, 0, 0),      # Blue
    'surprise': (0, 255, 255),# Yellow
    'neutral': (255, 255, 255),# White
    'No Face': (128, 128, 128), # Gray
    'Error': (0, 0, 0),      # Black
    'Unknown': (128, 128, 128) # Gray
}
default_color = (128, 128, 128) # Gray as default if emotion not in map
# --- Configuration ---
detector_backend = 'mtcnn'
analysis_interval = 0.25 # How often to run analysis (in seconds) - adjust as needed

# --- Shared Resources ---
frame_queue = queue.Queue(maxsize=1) # Queue to pass latest raw frame to worker
result_queue = queue.Queue(maxsize=1) # Queue to pass latest annotated frame back to main
stop_event = threading.Event() # Signal threads to stop

# --- Analysis Worker Thread Function ---
def analysis_worker():
    print("[ANALYSIS THREAD] Started.")
    last_analysis_time = 0
    
    while not stop_event.is_set():
        try:
            # Get the latest frame from the queue (non-blocking)
            raw_frame = frame_queue.get_nowait() 
        except queue.Empty:
            time.sleep(0.01) # Wait briefly if no frame is available
            continue # Skip if queue is empty

        current_time = time.time()
        # Only run analysis periodically
        if current_time - last_analysis_time < analysis_interval:
            # Put the raw frame back if we skip analysis this time, so display doesn't stall
            # (Optional: could just display frame without annotations)
            try:
                if result_queue.full(): result_queue.get_nowait()
                result_queue.put_nowait(raw_frame) 
            except queue.Full: pass
            time.sleep(0.01) 
            continue

        last_analysis_time = current_time
        annotated_frame = raw_frame.copy() # Work on a copy

        try:
            # Analyze the frame
            results = DeepFace.analyze(
                img_path=raw_frame, # Analyze the raw frame
                actions=['emotion'], 
                enforce_detection=False,
                detector_backend=detector_backend,
                silent=True
            )

            if results and isinstance(results, list) and len(results) > 0:
                # --- Replace the entire loop below ---
                for face_result in results:
                    # This block is indented once relative to 'for'
                    region = face_result.get('region') 
                    emotions = face_result.get('emotion') 
                    dominant_emotion = face_result.get('dominant_emotion') 

                    # Check if all necessary data was found for this face
                    if region and emotions and dominant_emotion: 
                        # --- This entire block MUST be indented once relative to 'if' ---
                        # (i.e., indented twice relative to 'for')
                        try: # Add inner try-except for safety during dict access/drawing
                            x, y, w, h = region['x'], region['y'], region['w'], region['h'] 
                            confidence = emotions.get(dominant_emotion, 0)

                            # Get color based on dominant emotion, use default if not found
                            color = emotion_colors.get(dominant_emotion, default_color) 

                            # Draw bounding box around the face using the emotion color
                            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)

                            # Prepare text label
                            text = f"{dominant_emotion} ({confidence:.2f})"

                            # Put text above the bounding box, using the emotion color
                            text_y = y - 10 if y - 10 > 10 else y + 10
                            cv2.putText(annotated_frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        except KeyError as ke:
                             print(f"[ANALYSIS THREAD] Warning: Missing key in region dict: {ke}")
                        except Exception as draw_e:
                             print(f"[ANALYSIS THREAD] Error during drawing: {draw_e}")
                    # --- End of block indented under 'if region and ...' ---
                # --- End of for loop ---
                
            if result_queue.full(): result_queue.get_nowait()
            result_queue.put_nowait(annotated_frame)

        # --- This is the except block for the main 'try' ---
        except Exception as e:
            # ... (Keep the detailed traceback printing here) ...
            print(f"[ANALYSIS THREAD] Error during analysis loop:")
            print(f"  Error Type: {type(e).__name__}")
            print(f"  Error Details: {e}")
            print("--- Traceback ---")
            traceback.print_exc() 
            print("-----------------")

            # If analysis fails, put the raw frame back so display doesn't stall
            try:
                if result_queue.full(): result_queue.get_nowait()
                result_queue.put_nowait(raw_frame) 
            except queue.Full: pass
            time.sleep(0.5) # Avoid spamming errors


# --- Main Thread (Capture & Display) ---
if __name__ == "__main__":
    
    # Start the analysis worker thread
    worker_thread = threading.Thread(target=analysis_worker, daemon=True)
    worker_thread.start()

    # Webcam Setup
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        stop_event.set() # Signal worker thread to stop
        exit()

    print("Starting real-time analysis... Press 'q' to quit.")
    prev_time = time.time()
    latest_processed_frame = None

    while not stop_event.is_set():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            stop_event.set()
            break

        # --- Pass frame to worker thread ---
        try:
            # Put the raw frame into the queue for the worker
            # If queue is full, remove the old one first (non-blocking)
            if frame_queue.full():
                frame_queue.get_nowait() 
            frame_queue.put_nowait(frame) 
        except queue.Full:
            # Should not happen often with maxsize=1 and get_nowait logic
            pass 

        # --- Get latest processed frame from worker ---
        try:
            # Get the latest frame with annotations (non-blocking)
            latest_processed_frame = result_queue.get_nowait() 
        except queue.Empty:
            # If no new processed frame, use the raw frame or last processed one
            display_frame = latest_processed_frame if latest_processed_frame is not None else frame

        if latest_processed_frame is not None:
             display_frame = latest_processed_frame

        # --- Display FPS ---
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # --- Display the frame ---
        cv2.imshow('Real-time Emotion Analysis (Press Q to Quit)', display_frame)

        # --- Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            stop_event.set() # Signal worker thread to stop
            break

    # --- Cleanup ---
    worker_thread.join(timeout=1.0) # Wait briefly for worker to finish
    cap.release()
    cv2.destroyAllWindows()
    print("Analysis stopped.")
