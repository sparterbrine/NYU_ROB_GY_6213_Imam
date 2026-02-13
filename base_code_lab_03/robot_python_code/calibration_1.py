# capture_calibration_images.py
import cv2 as cv
import os
import time

def main():
    # Create directory
    save_dir = 'calibration_images_webcam'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Open camera
    print("Opening camera...")
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera. Trying other indexes...")
        for i in range(1, 5):
            cap = cv.VideoCapture(i)
            if cap.isOpened():
                print(f"Found camera at index {i}")
                break
        else:
            print("No camera found!")
            return
    
    # Set resolution (optional)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "="*60)
    print("CALIBRATION IMAGE CAPTURE TOOL")
    print("="*60)
    print("\nInstructions:")
    print("  Press SPACE to capture an image")
    print("  Press ESC to quit")
    print("  Keep chessboard fully in frame")
    print("  Take photos from different angles")
    print("\nCurrent settings:")
    print(f"  Save location: {os.path.abspath(save_dir)}")
    print(f"  Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    print("\n" + "="*60)
    
    img_counter = 0
    last_capture_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Add overlay text
        display_frame = frame.copy()
        cv.putText(display_frame, f"Images: {img_counter}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(display_frame, "SPACE to capture | ESC to quit", (10, 70), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show feed
        cv.imshow('Calibration Image Capture', display_frame)
        
        key = cv.waitKey(1)
        
        if key % 256 == 27:  # ESC
            print("\nExiting...")
            break
            
        elif key % 256 == 32:  # SPACE
            # Debounce
            current_time = time.time()
            if current_time - last_capture_time > 0.5:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_name = f"{save_dir}/calib_{timestamp}_{img_counter:03d}.jpg"
                cv.imwrite(img_name, frame)
                print(f"Saved: {img_name}")
                img_counter += 1
                last_capture_time = current_time
    
    cap.release()
    cv.destroyAllWindows()
    
    print(f"\nSummary:")
    print(f"   Captured {img_counter} images")
    print(f"   Location: {os.path.abspath(save_dir)}")
    print(f"   Ready for calibration!")

if __name__ == "__main__":
    main()