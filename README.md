# Computer Vision - training a recognition model for the classification of various simple body exercises 

## Outline
This repository details my submission to the computer science module coursework for my Meng University of Portsmouth course. This research attempts to synthesise an exerise recognition model which could identify and classify various simple body exercises. In brief, the obtained raw data is prepared using various computer vision techniques and later used to train a machine learning recognition model through two classification algorithms: the Random Forest and Support Vector Machine (SVM).

The accuracies of the obtained models are compared against an isolated subset of the overall data. 
As a consequence of the lack of consistent data for the chosen topic, the project tangentually attempts to evaluate the impact of utilising inconsistent training environment (data points acquired from several datasets) on the accuracy of the prediction model.

The raw data used within this research is acquired from various existing researches. Namely, it is the research database presented in Berkeley Multimodal Human Action Database (MHAD) and a large public dataset provided by ROSE Lab at the Nanyang Technological
University, Singapore and its researchers.

Note: I have ommitted the data used for this research due to the its large size. The data complete raw data can be acquired from the above sources.

See paper for more information: [Paper](https://github.com/ogrruz/computer-vision/blob/main/paper.pdf)

## Directories

* ExtractingDepth && framesToVideo - Support classes for management of raw data acquired from the original datasets. It is worth noting that code contains various methods for the extraction of repetitions from recorded video data.  Much of these are deprecated due to the lack of access to a recording device with the ability to capture RGB-D (depth) data.

* testingFile - this class contains methods relating to the extraction of SIFT features and various functions for the display of processed depth frames for the analysis and inclusion in the paper.

* workflow - main file, a python jupyter notebook, goes through the process of data preparation, data processing and the training of the model with various classification algorithm. The overall results of this projects are acqruired here. 
