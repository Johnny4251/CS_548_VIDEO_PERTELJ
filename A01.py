# Author:       John Pertell
# Subject:      CS548 - Video Processing/Vision
# Date:         09.09.24
# Desc:         This program plays a video and saves each 
#               picture as a .png file in a specified directory.
#               This program also supports playing/saving video 
#               frames at a desired new_fps. 

import sys
import os
import cv2
import shutil
from pathlib import Path

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

def save_frames(all_frames, output_dir, basename, fps=30):

    """

    This function takes a list of frames from a video, an 
    output directory and a basename to save all frames as .png files. 
    The image frames will be found under the 'output_dir/basename' 
    directory. 
    
    The user will also be able to specify the fps if desired. 

    """

    # compute the folder path
    video_folder = basename + "_" + str(fps)
    output_path = os.path.join(output_dir, video_folder)

    # if directory already exists, recursively remove & replace it!
    if os.path.exists(output_path): shutil.rmtree(output_path)
    os.makedirs(output_path)

    # iterate through each frame
    frame_count = len(all_frames)
    for frame_idx in range(frame_count):
        # get current_frame from list
        frame = all_frames[frame_idx]

        # compute image name based on it's index
        image_name = "image_%07d.png" % frame_idx

        # write the image to output_path
        image_path = os.path.join(output_path, image_name)
        cv2.imwrite(image_path, frame)

def change_fps(input_frames, old_fps, new_fps):

    """
    
    This function takes in an input_frame list and it's old_fps.
    It then returns a new list of frames with the desired new_fps.

    """

    # get the old number of old & new_frames
    old_frame_cnt = len(input_frames)
    new_frame_cnt = int(new_fps*old_frame_cnt/old_fps)

    # a list to store the new frames
    output_frames = []

    # compute the new_frames to the output_frames
    # for the new_fps.
    for i in range(new_frame_cnt):
        frame_idx = int(old_fps*i/new_fps)
        output_frames.append(input_frames[frame_idx])

    return output_frames

def main():
    
    # new_fps < 0 => not being used
    new_fps = 30

    # get argument count-> verify that the correct arguments have been passed
    argc = len(sys.argv)
    if argc < 3:
        print("Usage: <input_video_path> <output_directory> <OPTIONAL: fps>")
        exit(1)
    
    # optional argument #3 => fps
    if argc > 3: new_fps = int(sys.argv[3])

    # get the input_video_path's core
    input_video_path = sys.argv[1]
    input_core = Path(input_video_path).stem

    # output_dir is 2nd arg
    output_dir = sys.argv[2]

    # load_video returns a list of each frame
    all_frames = load_video_as_frames(input_video_path)

    # ensure load_video_as_frames exectured correctly, if not exit(1)
    if all_frames is None: 
        print("Error: Could not load video frames.") 
        exit(1)
    
    # display frames
    display_frames(all_frames, "Input Video", fps=30)
    
    # compute the new output_frames->display the frames->save the frames
    output_frames = change_fps(all_frames, 30, new_fps)
    display_frames(output_frames, "Output Video", fps=30)
    save_frames(output_frames, output_dir, input_core, fps=new_fps)
    
    # save all_frames into output_dir/input_core
    #save_frames(all_frames, output_dir, input_core, fps=30)

if __name__ == "__main__": main()