import cv2
import os

def concatenate_ppm_to_video(ppm_folder, output_video_path, fps=30.0):
    # Get the list of .ppm files in the folder and sort them in ascending order
    ppm_files = [f for f in os.listdir(ppm_folder) if f.endswith('.ppm')]
    ppm_files.sort()

    # Read the first .ppm file to get dimensions
    first_frame = cv2.imread(os.path.join(ppm_folder, ppm_files[0]), cv2.IMREAD_COLOR)
    height, width, _ = first_frame.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each frame to the video file
    for ppm_file in ppm_files:
        ppm_path = os.path.join(ppm_folder, ppm_file)
        frame = cv2.imread(ppm_path, cv2.IMREAD_COLOR)
        out.write(frame)

    # Release the VideoWriter object
    out.release()

    print("Video created successfully.")

def divide_video_into_repetitions(input_video_path, output_folder, total_repetitions):
    capture = cv2.VideoCapture(input_video_path)

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_repetition = total_frames // total_repetitions

    repetition_folders = []
    for i in range(1, 6):
        repetition_folder = os.path.join(output_folder, f"repetition_{i}")
        os.makedirs(repetition_folder, exist_ok=True)
        repetition_folders.append(repetition_folder)

    repetition_count = 1
    frame_number = 0
    current_repetition_folder = repetition_folders[repetition_count - 1]

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame_number += 1

        # Determine the current repetition folder
        if frame_number > frames_per_repetition * repetition_count:
            repetition_count += 1
            current_repetition_folder = repetition_folders[repetition_count - 1]

        # Save the frame to the current repetition folder
        frame_filename = os.path.join(current_repetition_folder, f"frame_{frame_number:04d}.ppm")
        cv2.imwrite(frame_filename, frame)

    capture.release()
    print("Video divided into repetitions successfully.")

def extract_repetitions_by_start_and_end_frames(input_video_path, repetition_frames, output_folder):
    capture = cv2.VideoCapture(input_video_path)

    repetition_count = 1
    repetition_start_frame, repetition_end_frame = repetition_frames[0]

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if repetition_start_frame <= capture.get(cv2.CAP_PROP_POS_FRAMES) <= repetition_end_frame:
            # Write the frame to the repetition video file
            repetition_video_path = f"{output_folder}/repetition_{repetition_count}.avi"
            if repetition_count == 1:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                repetition_out = cv2.VideoWriter(repetition_video_path, fourcc, 30, (frame.shape[1], frame.shape[0]))

            repetition_out.write(frame)
        else:
            # Move to the next repetition
            repetition_count += 1
            if repetition_count <= len(repetition_frames):
                repetition_start_frame, repetition_end_frame = repetition_frames[repetition_count - 1]

    capture.release()
    if repetition_count > 1:
        repetition_out.release()

    print("Repetitions extracted to separate videos successfully.")

def convert_video_to_png(input_video_path, output_path):
    capture = cv2.VideoCapture(input_video_path)
    frame_number = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        # Save the frame to the current repetition folder
        frame_filename = os.path.join(output_path, f"frame_{frame_number:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_number += 1

    capture.release()
    print("Video converted to a series of pngs.")

# usage
recording_index = 'CW/data/other/Repetition_6'
ppm_folder = f'{recording_index}/all_frames'
output_video_path = f'{ppm_folder}/output_video.avi'
output_repsDir_path = f'{recording_index}/'

concatenate_ppm_to_video(ppm_folder, output_video_path)
convert_video_to_png(f'{ppm_folder}/output_video.avi', f'{recording_index}/all_frames_png/')


