# Author:       John Pertell
# Course:       CS548 - Video Processing/Vision
# Date:         11.13.24

import cv2
import numpy as np

# Function used to seperate box into segments.
# Segments are broken from top down, starting with 1
# and going until break_count.
def segment_box(bounding_box, break_count=3, pref_section=1):
    ymin, xmin, ymax, xmax = bounding_box
    height = ymax - ymin
    section_height = height // break_count

    new_ymin = ymin + section_height * (pref_section - 1)
    new_ymax = new_ymin + section_height

    return [new_ymin, xmin, new_ymax, xmax]

# Function is used for easy preprocessing the frame
def process_frame(frame):
    output = np.copy(frame)

    # YUV histogram equalization.
    output = cv2.cvtColor(output, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(output)
    Y_eq = cv2.equalizeHist(Y)
    yuv_eq = cv2.merge((Y_eq, U, V))
    output = cv2.cvtColor(yuv_eq, cv2.COLOR_YUV2BGR)
    
    return output

def track_doggo(video_frames, first_box, break_count=4, pref_section=2, ymin_scalar=.8, ymax_scalar=1.2, xmin_scalar = 1, xmax_scalar = 1.05):
    keep_first_frame_cnt = 4
    boxes = [first_box for _ in range(keep_first_frame_cnt)]
    hist_boxes = [first_box for _ in range(keep_first_frame_cnt)]

    segmented_box = segment_box(first_box, break_count=break_count, pref_section=pref_section)

    first_frame = process_frame(video_frames[0])
    first_frame_hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)

    ymin, xmin, ymax, xmax = segmented_box
    roi = first_frame_hsv[ymin:ymax, xmin:xmax]
    
    roi_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    for i in range(keep_first_frame_cnt, len(video_frames)):
        frame = process_frame(video_frames[i])
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        ymin, xmin, ymax, xmax = hist_boxes[-1]
        width = xmax - xmin
        height = ymax - ymin
        
        # do histogram back projection => camshift
        back_proj = cv2.calcBackProject([frame_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
        cv2.normalize(back_proj, back_proj, 0, 255, cv2.NORM_MINMAX)
        _, window = cv2.CamShift(back_proj, (xmin, ymin, width, height), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
        x, y, w, h = window

        # compute weighted average
        weight_prev = 0.5
        weight_new = 0.5
        avg_ymin = int(weight_prev * hist_boxes[-1][0] + weight_new * y)
        avg_xmin = int(weight_prev * hist_boxes[-1][1] + weight_new * x)
        avg_ymax = int(weight_prev * hist_boxes[-1][2] + weight_new * (y + h))
        avg_xmax = int(weight_prev * hist_boxes[-1][3] + weight_new * (x + w))

        box = [avg_ymin, avg_xmin, avg_ymax, avg_xmax]
        scaled_box = [int(box[0] * ymin_scalar), int(box[1] * xmin_scalar), int(box[2] * ymax_scalar), int(box[3] * xmax_scalar)]

        boxes.append(scaled_box)
        hist_boxes.append(box)

    return boxes
