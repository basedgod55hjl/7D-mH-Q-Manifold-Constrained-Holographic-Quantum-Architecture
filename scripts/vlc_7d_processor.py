# File: scripts/vlc_7d_processor.py
# Captures frames from a video or stream and pushes them to the 7D Crystal API bridge
import cv2
import requests
import base64
import time
import argparse

API_URL = "http://localhost:17777/vlc/process_frame"

def process_video_stream(video_path):
    print(f"🎬 Opening video stream: {video_path}")
    
    # Open the video stream (can be a local file or network stream from VLC)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    frame_id = 0
    start_time = time.time()
    
    print("🔮 Connecting to 7D Crystal Manifold API...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process one frame every ~30 frames to avoid overloading the stub API
            if frame_id % 30 == 0:
                # Resize for network efficiency if needed
                small_frame = cv2.resize(frame, (640, 360))
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', small_frame)
                
                # Convert to base64 string
                img_str = base64.b64encode(buffer).decode('utf-8')
                
                # Send to our 7D API
                payload = {
                    "frame_id": frame_id,
                    "width": 640,
                    "height": 360,
                    "data": img_str
                }
                
                try:
                    response = requests.post(API_URL, json=payload, timeout=2.0)
                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ Frame {frame_id} processed: {result.get('manifold_projections', 0)} projections in {result.get('time_ms', 0)}ms")
                    else:
                        print(f"⚠️ API Error: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    print("❌ Connection Error: Is the 7D Crystal Runtime API running on port 17777?")
                    break
                    
            frame_id += 1
            
            # Show the video being processed (optional)
            cv2.imshow('7D Crystal Input Stream', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Processing stopped by user.")
        
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    print(f"📊 Processed {frame_id} frames in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLC 7D Crystal Frame Processor")
    parser.add_argument("video", help="Path to video file or stream URL")
    
    args = parser.parse_args()
    process_video_stream(args.video)
