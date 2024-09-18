# This and the yolov3 files are needed to run dogvideo2

import deeplabcut

# Step 1: Create a new DeepLabCut project
# Replace with the actual video path of your dog walking video
video_path = ['Dog_videos/IMG_5260.MOV']

# Create a new DeepLabCut project (this will create a new project directory)
deeplabcut.create_new_project('DogGaitAnalysis', 'DogGaitTeam', video_path, copy_videos=True)

# Step 2: Manually label data (you will have to use the GUI for this step)
# After running the above, go into the created project folder and label frames for the pose estimation model.

# Step 3: Train the model after labeling
# Replace 'path_to_config.yaml' with the path to the config file in the newly created project directory
deeplabcut.train_network('DeepLabCutProject/config.yaml')

# Step 4: Evaluate the trained model to check performance
deeplabcut.evaluate_network('DeepLabCutProject/config.yaml')

# Once the model is trained, you can use it in the next file for dog pose estimation.