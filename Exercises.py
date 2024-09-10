import numpy as np
import cv2


"""
Exercises from: CS_490_548_01_VideoBasics - Dr. Michael J. Reale
"""
class VideoBasics:
    def gray_slice(image, minVal=100, maxVal=200, gray=True):
        output = np.copy(image)

        # Convert image from color to grayscale → output
        if gray:  output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # Use np.where to clamp the values:
        output = np.where(output <= minVal, minVal, output)
        output = np.where(output >= maxVal, maxVal, output)

        # Return output
        return output

    # MY EYES!!!
    def deres(image, scalar=10):
        # my personal twist adds a custom optional scalar
        if scalar < 1.0: scalar = 1.0

        output = np.copy(image)
        
        output = cv2.resize(output, dsize=(0,0), fx=1.0 / scalar, fy=1.0 / scalar, interpolation=cv2.INTER_LINEAR)
        output = cv2.resize(output, dsize=(0,0), fx=1.0 * scalar, fy=1.0 * scalar, interpolation=cv2.INTER_NEAREST)

        return output
    
    def blur_across_buffer(frame_buffer, time_index, frame, a_ghost=False):
        """
        In the main function:
        Before the loop, create a frame_buffer of zeroes with 10 frames + same shape as
        the images, and a dtype of np.float64
        Set time_index to 0
        In the loop, call blur_across_buffer() to get our processed image
        """

        # Convert frame to dtype np.float64 and scale to [0,1]
        frame = frame.astype(np.float64)
        frame /= 255.0

        # Write the frame to the current time_index location in frame_buffer
        frame_buffer[time_index] = frame

        # Get the mean across time
        avg_idx = np.mean(frame_buffer, axis=0)

        # Increment time_index, modding by the number of frames in frame_buffer
        time_index += 1
        time_index %= len(frame_buffer)

        # A shortcut to the next exercise.. if desired
        if a_ghost:
            fimage = a_ghost(frame, avg_idx)
            return fimage

        # Return frame_buffer, time_index, and the average frame
        return frame_buffer, time_index, avg_idx
        

    def a_ghost(frame, avg_idx):
        """
        Use this in conjunction w/blur_across_buffer
        """
        fimage = np.copy(frame)

        # Convert the frame into a np.float64 image of scale [0,1] → fimage
        fimage = frame.astype(np.float64) / 255.0

        # SUBTRACT the mean image to get the processed image
        fimage = np.absolute(fimage - avg_idx) # avg_idx comes from blur_across_buffer return

        return fimage

class CustomFun:
    def threshold_channel(image, threshold_value, channel=None):
        if channel is not None and len(image.shape) == 3:
            channel_data = image[:, :, channel]
            thresholded_channel = np.where(channel_data >= threshold_value, 255, 0).astype(np.uint8)
            
            thresholded_image = image.copy()
            thresholded_image[:, :, channel] = thresholded_channel
        else:
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
            
            thresholded_image = np.where(gray_image >= threshold_value, 255, 0).astype(np.uint8)

        return thresholded_image