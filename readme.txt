                    PPE Detector README

                            Overview
This project implements a Personal Protective Equipment (PPE) detection system using 
YOLOv8 for equipment detection and FaceNet for face recognition. The system identifies 
individuals who are not wearing the necessary PPE and highlights them with a red
bounding box. It also provides a list of missing equipment and the name of the employee.

                         File Structure
The project contains the following files and directories:

/PPE_Detector
│
├── Model                           # Contains files and subfolders necessary for model training, including scripts, pre-trained models, and configuration files.
├── embedding_np                    # Python source file for embeddings for facial recognition
├── Equip_detect                    # Python source file for equipment detection
├── Facialembedding.npz             # Pre-trained facial embedding data
├── glasses                         # Python source file for detecting glasses
├── main                            # main file for execytion of demo
├── model_inference_imgs            # Python script t infer model on images
├── face_recognition                # Python source file for making predictions
├── person.pkl                      # Pre-trained model file for person recognition
├── model2.pkl                      # Pre-trained model file for facial recognition
└── readme                          # This README file

                            Requirements
To run this project, ensure you have the following libraries installed:

Ultralytics
MTCNN
SciKit-Learn
OpenCV
NumPy

You can install the necessary packages using pip:

pip install opencv-python numpy Scikit-Learn MTCNN Ultralytics

                               Setup
Clone the repository or download the files to your local machine.
Ensure that you have the required model files in the project directory.
Place the demo videos in the Videos_demo folder for testing.
                              
                               Usage
To run the PPE detection system, execute the main script in the Equip_detect directory

                           Functionality
The system processes video input and detects faces using the FaceNet model.
It checks for the presence of required PPE (e.g., helmets, glasses, masks).
If PPE is missing, a red bounding box is drawn around the individual, displaying:
The employee's name
A list of missing equipment
Example Output
The output will show a video stream with bounding boxes around individuals lacking proper PPE, 
along with their names and missing equipment listed above the box.

                            Contributing
If you'd like to contribute to this project, please fork the repository and submit a pull
request with your enhancements or bug fixes.

contact
For any queries or further discussion feel free to contact at mspl@uettaxila.edu.pk
