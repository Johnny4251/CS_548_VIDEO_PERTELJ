import sys
import numpy as np
import time
import cv2
from Exercises import VideoBasics, CustomFun

def display_file(filename, looping=True):
    capture = cv2.VideoCapture(filename)
    
    if not capture.isOpened():
        print("ERROR: Could not open or find the video!")
        exit(1)

    windowName = f"{filename}"
    cv2.namedWindow(windowName)

    key = -1
    while key == -1:
        ret, frame = capture.read()
        if ret == True:   
            cv2.imshow(windowName, frame)

            # in order to loop a video...
            if looping:
                # Get frame count and current frame index
                frame_cnt = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_idx = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

                # If at last frame, wrap current frame(frame_idx) from end(frame_cnt)->start(0)
                if (frame_idx == frame_cnt): 
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            print("Something bad happened...") 
            break
        key = cv2.waitKey(30)

    capture.release()

def display_video():
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not capture.isOpened():
        print("ERROR: Cannot open capture!")
        exit(1)
        
    start_time = time.time()
    windowName = f"Webcam {int(start_time)}"
    key = -1
    while key == -1:
        ret, frame = capture.read()
        #frame = VideoBasics.deres(frame, scalar=(10 / (time.time() - start_time)))
        if ret == True:        
            cv2.imshow(windowName, frame)
            cv2.setWindowTitle(windowName, f"Webcam {int(time.time()) - int(start_time)}")
        else: break

        key = cv2.waitKey(30)

    capture.release()

def main():        
    if len(sys.argv) <= 1:
        display_video()
    else:
        filename = sys.argv[1]
        display_file(filename)

    cv2.destroyAllWindows()


if __name__ == "__main__": 
    main()