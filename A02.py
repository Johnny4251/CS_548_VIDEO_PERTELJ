# Author:       John Pertell
# Course:       CS548 - Video Processing/Vision
# Date:         10.06.24

import A01
import numpy as np
import cv2
from enum import Enum 

class OPTICAL_FLOW(Enum): 
    HORN_SHUNCK = "horn_shunck" 
    LUCAS_KANADE = "lucas_kanade"

def compute_video_derivatives(video_frames, size):

    """

    This function gets and returns the derivatives of the given
    video. 

    """

    if size == 2:
        kfx = np.array([[-1, 1], [-1, 1]])
        kfy = np.array([[-1, -1], [1, 1]])
        kft1 = np.array([[-1, -1], [-1, -1]])
        kft2 = np.array([[1, 1], [1, 1]])
    elif size == 3:
        kfx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kfy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        kft1 = np.array([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
        kft2 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    else: return None

    previous_frame = None
    all_fx = []
    all_fy = []
    all_ft = []

    for frame in video_frames:
        
        # Convert the image to a grayscale, float64 image with range [0,1] 
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = gray_frame.astype(np.float64)
        gray_frame /= 255.0

        # If the previous frame is not set, set it to the current frame
        if previous_frame is None:
            previous_frame = gray_frame

        # Apply the filters to get the fx, fy and ft values across both frames
        fx = cv2.filter2D(previous_frame, -1, kfx) + cv2.filter2D(gray_frame, -1, kfx)
        fy = cv2.filter2D(previous_frame, -1, kfy) + cv2.filter2D(gray_frame, -1, kfy)
        ft = cv2.filter2D(previous_frame, -1, kft1) + cv2.filter2D(gray_frame, -1, kft2)

        # Scale the results accordingly
        if size == 2:
            fx /= 4.0
            fy /= 4.0
            ft /= 4.0
        else:
            fx /= 8.0
            fy /= 8.0
            ft /= 16.0

        # Keep a running count of all fx,fy & ft
        all_fx.append(fx)
        all_fy.append(fy)
        all_ft.append(ft)

        # Set previous frame to current frame
        previous_frame = gray_frame

    # Return three lists: all_fx, all_fy, all_ft
    return all_fx, all_fy, all_ft

def compute_one_optical_flow_horn_shunck(fx, fy, ft, max_iter, max_error, weight=1.0):
    """

    Compute one optical flow using the Horn-Shuck method.

    Most of the code is taken from 'ProfExercises02.py'. Modified
    to fit requirements of assign02.

    """

    u = np.zeros(fx.shape, dtype="float64")
    v = np.zeros(fx.shape, dtype="float64")

    lap_filter = np.array([[0, 0.25, 0],
                           [0.25, 0, 0.25],
                           [0, 0.25, 0]], dtype="float64")
    
    iter_cnt = 0
    converged = False
    
    while not converged:
        uav = cv2.filter2D(u, cv2.CV_64F, lap_filter)
        vav = cv2.filter2D(v, cv2.CV_64F, lap_filter)

        P = fx*uav + fy*vav + ft
        D = weight + fx*fx + fy*fy

        PD = P/D

        u = uav - fx*PD
        v = vav - fy*PD

        # Get the AVERAGE of the error
        error = np.mean(np.abs(PD))
        
        iter_cnt += 1
        
        if error <= max_error or iter_cnt >= max_iter:
            converged = True

    extra = np.zeros_like(u)
    combo = np.stack([u, v, extra], axis=-1)

    return combo, error, iter_cnt


def compute_one_optical_flow_lucas_kanade(fx, fy, ft, win_size):
    """

    Compute the optical flow using the Lucas-Kanade method for each block.
    Technique is from slide 102 of 'CS_490_548_02_OpticalFlow'.

    """
    height, width = fx.shape
    u = np.zeros((height, width), dtype="float64")
    v = np.zeros((height, width), dtype="float64")
    
    for y in range(0, height, win_size):
        for x in range(0, width, win_size):
            # Get block fx,fy,ft
            block_fx = fx[y:y+win_size, x:x+win_size]
            block_fy = fy[y:y+win_size, x:x+win_size]
            block_ft = ft[y:y+win_size, x:x+win_size]

            # Compute useful terms
            sum_fx_fy = np.sum(block_fx * block_fy)
            sum_fx_ft = np.sum(block_fx * block_ft) 
            sum_fy_ft = np.sum(block_fy * block_ft)
            sum_fx2 = np.sum(block_fx**2)
            sum_fy2 = np.sum(block_fy**2)

            # Compute u & v fraction numerators & denominators using 'useful terms'
            u_numerator = (sum_fy2 * sum_fx_ft) - (sum_fx_fy * sum_fy_ft)
            v_numerator = (sum_fx2 * sum_fy_ft) - (sum_fx_fy * sum_fx_ft)
            denominator = (sum_fx2 * sum_fy2) - (sum_fx_fy**2)

            # If the denominator is less than 1e-6, leave the u and v values at zero. 
            if denominator < 1e-6:
                u[y:y+win_size, x:x+win_size] = 0
                v[y:y+win_size, x:x+win_size] = 0
                continue

            u_val = u_numerator / denominator
            v_val = v_numerator / denominator
            
            u[y:y+win_size, x:x+win_size] = u_val
            v[y:y+win_size, x:x+win_size] = v_val

    extra = np.zeros_like(u)
    optical_flow = np.stack([u, v, extra], axis=-1)

    return optical_flow

def compute_optical_flow(video_frames, method=OPTICAL_FLOW.HORN_SHUNCK, max_iter=10, max_error=1e-4, horn_weight=1.0, kanade_win_size=19):
    """
    
    Compute Optical Flow, given a method from OPTICAL_FLOW enum. 

    """
    
    flow_frames = []

    if method == OPTICAL_FLOW.HORN_SHUNCK:
        window_size = 2
    elif method == OPTICAL_FLOW.LUCAS_KANADE:
        window_size = 3
    else: return None

    fx, fy, ft = compute_video_derivatives(video_frames, window_size)

    # Iterate through each frame in video_frames
    for i in range(len(video_frames)):
        if method == OPTICAL_FLOW.HORN_SHUNCK:
            optical_flow, error, iter_cnt = compute_one_optical_flow_horn_shunck(
                                            fx[i], fy[i], ft[i], 
                                            max_iter, max_error, horn_weight )
        elif method == OPTICAL_FLOW.LUCAS_KANADE:
            optical_flow = compute_one_optical_flow_lucas_kanade(
                                                                fx[i], 
                                                                fy[i],
                                                                ft[i], 
                                                                win_size=kanade_win_size)
        flow_frames.append(optical_flow)

    return flow_frames

def main():     

    """

    Main function example is from 'CS_490_548_Assign02.pdf'.

    """

    # Load video frames 
    video_filepath = "assign02/input/simple/image_%07d.png" 
    #video_filepath = "assign02/input/noice/image_%07d.png" 
    video_frames = A01.load_video_as_frames(video_filepath) 
     
    # Check if data is invalid 
    if video_frames is None: 
        print("ERROR: Could not open or find the video!") 
        exit(1) 
         
    # Calculate optical flow 
    flow_frames = compute_optical_flow(video_frames, method=OPTICAL_FLOW.LUCAS_KANADE) 
    
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
    