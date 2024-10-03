import A01
import numpy as np
import cv2
from enum import Enum 

class OPTICAL_FLOW: 
    HORN_SHUNCK = "horn_shunck" 
    LUCAS_KANADE = "lucas_kanade"

def compute_video_derivatives(video_frames, size):
    pass

def compute_one_optical_flow_horn_shunck(fx, fy, ft, max_iter, max_error, weight=1.0):
    pass

def compute_optical_flow(video_frames, method=OPTICAL_FLOW.HORN_SHUNCK, max_iter=10, max_error=1e-4, horn_weight=1.0, kanade_win_size=19):
    pass


def main():     

    """
    Code from 'CS_490_548_Assign02.pdf'
    """

    # Load video frames 
    video_filepath = "assign02/input/simple/image_%07d.png" 
    #video_filepath = "assign02/input/noice/image_%07d.png" 
    video_frames = A01.load_video_as_frames(video_filepath) 
     
    # Check if data is invalid 
    if video_frames is None: 
        print("ERROR: Could not open or find the video!") 
        exit(1) 
         
    # OPTIONAL: Only grab the first five frames 
    video_frames = video_frames[0:5] 
         
    # Calculate optical flow 
    flow_frames = compute_optical_flow(video_frames, method=OPTICAL_FLOW.HORN_SHUNCK) 
 
    # While not closed... 
    key = -1 
    ESC_KEY = 27 
    SPACE_KEY = 32 
    index = 0 
     
    while key != ESC_KEY: 
        # Get the current image and flow image 
        image = video_frames[index] 
        flow = flow_frames[index] 
         
        flow = np.absolute(flow) 
         
        # Show the images 
        cv2.imshow("ORIGINAL", image) 
        cv2.imshow("FLOW", flow) 
             
        # Wait 30 milliseconds, and grab any key presses 
        key = cv2.waitKey(30) 
         
        # If space, move forward 
        if key == SPACE_KEY: 
            index += 1 
            if index >= len(video_frames): 
                index = 0 
 
    # Destroy the windows     
    cv2.destroyAllWindows() 
     
if __name__ == "__main__":  
    main() 
    