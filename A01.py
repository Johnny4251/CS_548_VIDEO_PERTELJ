import sys
import numpy as np
import cv2

def load_video_as_frames(video_filepath):

    """

    This function takes in a filepath string for a video and attempts 
    to open the file. In the event that the video is not found,
    this function will return 'None' and print an error.

    The function will then store all of the video's frames into
    a list and finish by returning the list.

    """

    # open video
    capture = cv2.VideoCapture(video_filepath)
        
    # if video cannot be opened..
    if not capture.isOpened():
        print("ERROR: Could not open or find the video!")
        return None
    
    # keep track of frames as a list
    video_frames = []

    # looping through all frames
    while True:

        # find frame
        ret, frame = capture.read()
        
        # append each frame to list
        if ret: video_frames.append(frame)

        # release the video, return the list of frames
        else:
            capture.release()
            return video_frames


def compute_wait(fps):
    """

    Computes the wait time in between frames as an int.
    Given an fps(frames per second) value.

    """
    return int(1000.0/fps)

def display_frames(all_frames, title, fps=30):
    
    """
    
    Displays each frame at a desired fps. The window title
    is determined by the 'title' string argument.

    """

    # compute the wait_time from fps
    wait_time = compute_wait(fps)

    # display each frame at the desired fps
    frame_count = len(all_frames)
    for frame_idx in range(frame_count):

        # get current_frame from list->show frame
        frame = all_frames[frame_idx]
        cv2.imshow(title, frame)

        # wait_time is in ms
        cv2.waitKey(wait_time)

    # cleanup window
    cv2.destroyWindow(title)


def main():
    """
    
    Main loop, contents are not intended for prod.. yet. 
    Just for testing functions.
    
    """
    
    frames = load_video_as_frames("assign01/input/noice.mp4")
    display_frames(frames, "Title", fps=20)
    

if __name__ == "__main__":
    main()