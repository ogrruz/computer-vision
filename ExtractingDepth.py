import cv2
import numpy as np
import glob

def preprocess_depth_frames(depth_frames):
    # Add any necessary preprocessing steps here
    # For example, you can normalize, threshold, or scale the depth values
    # If the depth_frames are already in the correct format, you may skip this step
    # Return the preprocessed depth_frames
    return depth_frames

def save_depth_video(file_name, depth_frames):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_file_path = f'C:/Users/Garry/Documents/ComputerVision/CW/data/{exercise}/{file_name}_depth.avi'
    frame_width = depth_frames.shape[2]
    frame_height = depth_frames.shape[1]
    fps = 30  # Adjust the frames per second as needed

    # Create VideoWriter object
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (frame_width, frame_height))

    # Save each depth frame to the video file
    for frame in depth_frames:
        out.write(frame)

    # Release the VideoWriter
    out.release()

exercise = "crunches"   # "crunches", "jumping_jacks", "push_up", "other" 

# Get the file paths of the .avi files
avi_files = [file for file in glob.glob(f'./data/{exercise}/*.avi')]

# Process each .avi file
for file in avi_files:
    # Extract the name without the extension
    file_name = file.split('/')[-1].split('.')[0]

    # Load the depth frames from the .avi file
    capture = cv2.VideoCapture(file)
    depth_frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        depth_frames.append(frame)
    capture.release()
    depth_frames = np.array(depth_frames)

    # Preprocess the depth frames (if needed)
    preprocessed_depth_frames = preprocess_depth_frames(depth_frames)

    # Perform your tasks on the preprocessed depth data
    # For example, feature extraction, classification, etc.

    # Optionally, save the preprocessed depth frames as a new video file
    save_depth_video(file_name, preprocessed_depth_frames)

    ans = int(input("Continue? 1-yes, 2-no: "))
    if ans == 2:
        break

print("Depth data processing completed.")
