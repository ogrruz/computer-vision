import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

all_descriptors = []
#data_point_labels = []
dataset_bow_descriptors = []
dataset_bow_labels = []

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass an empty dictionary

matcher = cv2.FlannBasedMatcher(index_params, search_params)

cluster_size = 20  # 20
tc = (cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)
retries = 3
flags = cv2.KMEANS_PP_CENTERS

bow_trainer = cv2.BOWKMeansTrainer(cluster_size, tc, retries, flags)

sift = cv2.SIFT_create(nfeatures=100)
bow_descriptor_extractor = cv2.BOWImgDescriptorExtractor(sift, matcher)

import cv2
import numpy as np

THRESHOLD_VALUE = 5
MHI_DURATION = 1

def load_and_preprocess_colour_images(depth_folder):
    depth_images = []
    for file in glob.glob(os.path.join(depth_folder, '*.ppm')):
        depth_image = cv2.imread(file, cv2.IMREAD_UNCHANGED)

        # Preprocess the depth_image
        #...
        # Convert the depth_image to 8-bit unsigned integer
        depth_image = cv2.convertScaleAbs(depth_image)

    
        depth_images.append(depth_image)

    return depth_images

def test_masking(depth_frame):
    depth_image = cv2.imread(depth_frame, cv2.IMREAD_UNCHANGED)
    cv2.imshow("16bit depth", depth_image)
    masked = mask_and_normalise(depth_image)
    cv2.imshow("16bit masked", masked)

    eight_bit_depth_image = cv2.convertScaleAbs(masked)
    denoised_depth_image = cv2.medianBlur(eight_bit_depth_image, 9)
    cv2.imshow("8bit masked", denoised_depth_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def filter_mask(depth_frame, desired_shade):
#     # Apply a threshold to create a binary mask
#     _, binary_mask = cv2.threshold(depth_frame, desired_shade, 255, cv2.THRESH_BINARY)

#     # Optional: Apply morphological operations like dilation to further process the mask
#     kernel = np.ones((5, 5), np.uint8)
#     binary_mask = cv2.dilate(binary_mask, kernel, iterations=3)

#     # Return the filtered mask 
#     return binary_mask

def filter_mask(depth_frame, desired_grey, tolerance=20):

    # pre processing gaussian blur
    #filtered_image = cv2.GaussianBlur(depth_frame, (5, 5), 0)

    # Create a binary mask for the desired shade of grey
    lower_grey = max(0, desired_grey - tolerance)
    upper_grey = min(255, desired_grey + tolerance)
    mask = cv2.inRange(depth_frame, lower_grey, upper_grey)  

    # Apply the mask to the depth frame using bitwise AND operation
    masked_depth_frame = cv2.bitwise_and(depth_frame, depth_frame, mask=mask)

    return masked_depth_frame

def mask_and_normalise(depth_frame):
    threshold_value = 0  

    # Create a binary mask based on the threshold value
    mask = (depth_frame > threshold_value).astype(np.uint8) * 255

    # Apply the mask to the depth frame using bitwise AND operation
    masked_depth_frame = cv2.bitwise_and(depth_frame, depth_frame, mask=mask)

    # Normalize the masked depth frame to the 8-bit range (0 to 255)
    normalized_depth_frame = cv2.normalize(masked_depth_frame, None, 0, 255, cv2.NORM_MINMAX)

    #focus mask
    normalized_depth_frame = filter_mask(normalized_depth_frame, 85)

    return normalized_depth_frame

def load_and_preprocess_depth_images(depth_folder):
    depth_images = []
    for file in glob.glob(os.path.join(depth_folder, '*.pgm')):
        depth_image = cv2.imread(file, cv2.IMREAD_UNCHANGED)

        # # Preprocess the depth_image
        # masked_depth_image = mask_and_normalise(depth_image)
        # # Convert the depth_image to 8-bit unsigned integer
        # #depth_image = cv2.convertScaleAbs(depth_image)
        # depth_image = cv2.convertScaleAbs(masked_depth_image)

        # Preprocess the depth_image
        masked_depth_image = mask_and_normalise(depth_image)
        # Convert the depth_image to 8-bit unsigned integer
        #depth_image = cv2.convertScaleAbs(depth_image)
        depth_image = cv2.convertScaleAbs(masked_depth_image)
        denoised_depth_image = cv2.medianBlur(depth_image, 9)

    
        depth_images.append(denoised_depth_image)

    # Convert the list of depth images to a numpy array
    #depth_images_array = np.array(depth_images)

    return depth_images

#extracts sift features from the total data for an exercise 
def extract_sift_from_exercise(exercise_folder, exercise_label):

    # Get a list of subdirectories (repetition folders) in exercise_folder
    repetition_folders = os.listdir(exercise_folder)
    print(repetition_folders)

    repcount = 0

    # Process all rep samples
    for repetition_folder in repetition_folders:   #glob.glob(os.path.join(exercise_folder, '*/')):
        depth_folder = os.path.join(exercise_folder, repetition_folder, 'depth')
        depth_images = load_and_preprocess_depth_images(depth_folder)
        
        #process 1 rep
        for depth_image in depth_images:
            
            keypoint, descriptor = sift.detectAndCompute(depth_image, None)

            if descriptor is not None and descriptor.shape[0] > 0:
                all_descriptors.append(descriptor)
                bow_trainer.add(descriptor)
            else:
                print("No descriptors extracted from this repetition",)
            if repcount == 0:
                image_with_keypoints = cv2.drawKeypoints(depth_image, keypoint, None, (0, 0, 255), flags=0)
                cv2.imshow('Keypoints', image_with_keypoints)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            print("pgm file processed: ")
        repcount += 1
        print("rep processed: ", repcount)

    print("Exercise class processed:", exercise_label)


def extract_sift_colour_from_exercise(exercise_folder, exercise_label):

    # Get a list of subdirectories (repetition folders) in exercise_folder
    repetition_folders = os.listdir(exercise_folder)
    print(repetition_folders)

    repcount = 0

    # Process all rep samples
    for repetition_folder in repetition_folders:   #glob.glob(os.path.join(exercise_folder, '*/')):
        depth_folder = os.path.join(exercise_folder, repetition_folder, 'colour')
        depth_images = load_and_preprocess_colour_images(depth_folder)
        
        #process 1 rep
        for depth_image in depth_images:
            keypoint, descriptor = sift.detectAndCompute(depth_image, None)

            if descriptor is not None and descriptor.shape[0] > 0:
                all_descriptors.append(descriptor)
                bow_trainer.add(descriptor)
            else:
                print("No descriptors extracted from this repetition",)
            if repcount == 9:
                image_with_keypoints = cv2.drawKeypoints(depth_image, keypoint, None, (0, 255, 0), flags=0)
                cv2.imshow('Keypoints', image_with_keypoints)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            print("pgm file processed: ")
        repcount += 1
        print("rep processed: ", repcount)

    print("Exercise class processed:", exercise_label)

def extract_siftmhi_from_exercise(exercise_folder, exercise_label):

    # Get a list of subdirectories (repetition folders) in exercise_folder
    repetition_folders = os.listdir(exercise_folder)
    print(repetition_folders)

    repcount = 0

    # Process all rep samples
    for repetition_folder in repetition_folders:   #glob.glob(os.path.join(exercise_folder, '*/')):
        depth_folder = os.path.join(exercise_folder, repetition_folder, 'depth')
        depth_images = load_and_preprocess_depth_images(depth_folder)
        
        #process 1 rep
        for depth_image in depth_images:

            mhi = generate_mhi(depth_images)

            
            keypoint, descriptor = sift.detectAndCompute(mhi, None)

            if descriptor is not None and descriptor.shape[0] > 0:
                all_descriptors.append(descriptor)
                bow_trainer.add(descriptor)
            else:
                print("No descriptors extracted from this repetition",)
            if repcount == 9:
                image_with_keypoints = cv2.drawKeypoints(mhi, keypoint, None, (0, 0, 255), flags=0)
                cv2.imshow('Keypoints', image_with_keypoints)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            print("pgm file processed: ")
        repcount += 1
        print("rep processed: ", repcount)

    print("Exercise class processed:", exercise_label)

def generate_mhi(frames, threshold_value=0.05, mhi_duration=1):
    number_of_frames = len(frames)
    height, width = frames[0].shape

    # Initialize the MHI accumulator
    mhi_accumulator = np.zeros((height, width), dtype=np.float32)

    for i in range(1, number_of_frames):
        current_frame = frames[i]

        # Calculate the difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(current_frame, frames[i - 1])

        # Convert the frame difference to a binary image based on the threshold value
        _, binary_diff = cv2.threshold(frame_diff, threshold_value, 1, cv2.THRESH_BINARY)

        # Update the MHI accumulator
        mhi_accumulator[binary_diff == 1] = i

        # Decrease the MHI values for old pixels
        mhi_accumulator[mhi_accumulator > 0] -= 1

        # Threshold the MHI to remove old values beyond the duration
        mhi_accumulator[mhi_accumulator < i - mhi_duration] = 0

    # Normalize the MHI to the range [0, 255] for visualization
    mhi_normalized = (mhi_accumulator / mhi_duration * 255).astype(np.uint8)

    return mhi_normalized

# SIFT EXTRACTION AND BOW REPRESENTATION
j_jacks_path = 'C:/Users/Garry/Documents/ComputerVision/CW/data/j_jacks'
jumping_path = 'C:/Users/Garry/Documents/ComputerVision/CW/data/jumps'
lateral_raise_path = 'C:/Users/Garry/Documents/ComputerVision/CW/data/lateral_raise'
other_path_path = 'C:/Users/Garry/Documents/ComputerVision/CW/data/other'

# images = load_and_preprocess_depth_images(j_jacks_path)    
# keypoints, descriptors = sift.detectAndCompute(images[0], None)
# print(descriptors)
test2 = 'C:/Users/Garry/Documents/ComputerVision/CW/data/j_jacks/'
extract_sift_from_exercise(test2 , "jumping_jack")
#extract_sift_from_exercise(jumping_path, "jump")
#extract_sift_from_exercise(lateral_raise_path, "lateral_raise")
#extract_sift_and_bow_from_exercise(other_path_path, "other")
#extract_sift_colour_from_exercise(j_jacks_path, "jumping_jack")

#print(len(dataset_bow_labels))
# dictionary = bow_trainer.cluster()

normal_frame = 'C:/Users/Garry/Documents/ComputerVision/CW/raw_data/a2-jumping_jacks/R01/all_frames/kin_k01_s01_a02_r01_depth_00150.pgm'
normal_frame2 = 'C:/Users/Garry/Documents/ComputerVision/CW/raw_data/a2-jumping_jacks/R03/all_frames/kin_k01_s01_a02_r03_depth_00154.pgm'
normal_frame_jump = 'C:/Users/Garry/Documents/ComputerVision/CW/raw_data/a1-jumping/R01/all_frames/kin_k01_s12_a01_r01_depth_00100.pgm'
normal_frame2_jump = 'C:/Users/Garry/Documents/ComputerVision/CW/raw_data/a1-jumping/R03/all_frames/kin_k01_s12_a01_r03_depth_00140.pgm'
normal_frame_lat = 'C:/Users/Garry/Documents/ComputerVision/CW/data/lateral_raise/Repetition_8/depth/kin_k01_s12_a05_r02_depth_00130.pgm'
other_frame = 'CW/data/other/Repetition_6/depth/kin_k01_s03_a07_r01_depth_00028.pgm'
depth_frame = cv2.imread(normal_frame, cv2.IMREAD_UNCHANGED)
test = 'C:/Users/Garry/Documents/ComputerVision/CW/data/j_jacks/Repetition_6/depth'


# test_masking(test)
# yuh = load_and_preprocess_depth_images('C:/Users/Garry/Documents/ComputerVision/CW/data/j_jacks/Repetition_6/depth')
# mhi = generate_mhi(yuh)
# cv2.imshow('mhi', mhi)
# cv2.waitKey(0)

# cv2.destroyAllWindows()