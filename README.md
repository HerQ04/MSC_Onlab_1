
# msc-projectlab1
Project Laboratory 1 for the BME MSc<br>
Image processing and machine learning to detect unauthorized access to warehouses/any observed areas
# Project contents
## YOLOv5 Model Trainer
**1. Prerequisites**

Download yolov5 and install required PIP modules.
Install RoboFlow and TensorBoard for dataset handling and result analyzing.
Open a command prompt in administrator mode where you want to install YOLOv5 and run the following commands:

    git clone -q https://github.com/ultralytics/yolov5
	cd yolov5
	pip install -qr requirements.txt
	pip install -q roboflow
	pip install -q tensorboard

 **2.1. Roboflow project setup**

To setup a Roboflow project, the [projectsetup.py](https://github.com/sommys/msc-projectlab1/blob/main/ModelTrainer/projectsetup.py) script and run it from a command line in administrator mode.
The script has the following arguments:
|Argument Name|Description|Default Value|
|:-:|-|:-:|
|workspace|the Roboflow workspace containing the project|msc-onlab-1|
|project|the id of the Roboflow project|person-so5ko|
|version|the version of the dataset|4|
|yoloversion|the version of YOLO to be used for the project setup (5 or 8)|5|

 **2.2. Model training**

Download the [train_model.py](https://github.com/sommys/msc-projectlab1/blob/main/ModelTrainer/train_model.py) script and run it from a command line in administrator mode.
The script has the following arguments:
|Argument Name|Description|Default Value|
|:-:|-|:-:|
|img|training and validating image size (in pixels)|640|
|batch|total batch size for all GPUs, use -1 for autobatch|auto|
|epochs|total training epochs|100|
|model|model to be used for training (shall be inside the models subfolder of the YOLOv5 folder)|yolov5m
|name|name of the current project output|exp|
|dataset|name of the dataset|person-4|
|workspace|the Roboflow workspace containing the project|msc-onlab-1|
|project|the id of the Roboflow project|person-so5ko|
|version|the version of the dataset|4|
|outputdir|the directory where the training results shall be saved|C:/yolov5|
|yolov8|flag for using yolov8|false|

Example:

	train_model.py --img 480 --batch 8 --epochs 300 --model yolov5x --name LargeModelTraining --dataset "person-1"

The dataset parameter shall be used if you have already setup a Roboflow project.
If you haven't already setup one, you can use the workspace, project and version arguments to run the setup script automatically before training as described in 2.1.

The script assumes that the path to the YOLOv5 repository is set in the YOLOV5_PATH system environment variable, or it is installed under
	
	C:\yolov5

or in case you are using YOLOv8, you should install it beforehand using the following command:
	
	pip install ultralytics

**2.3. Analyzing results**

Download the [results.py](https://github.com/sommys/msc-projectlab1/blob/main/ModelTrainer/results.py) script and run it from a command line in administrator mode.
The script runs automatically at the end of training as well, but you can use this script in case you want to analyze results without training.
The script has the following arguments:
|Argument Name|Description|Default Value|
|:-:|-|:-:|
|yoloversion|the version of YOLO to be used for the handling of the results (5 or 8)|5|
|resultdir|the directory containing the results|C:\YOLOtraining|


For showing the results you should have Google Chrome installed under

    C:\Program Files\Google\Chrome\Application
If you have it installed somewhere else add the following environment variable for your system:

    CHROME_PATH -> path\to\chrome

## PersonDetector
Python script that allows using the previously trained YOLOv5 models on images, videos and webcam sources to detect people.

**1. Prerequisites**
Install PyTorch, OpenCV and download YOLOv5 and model weights of your choice (or use the previously listed model trainer).

Used environment variables during the script:
|Name|Description|Default value|
|:-:|-|:-|
|YOLOV5_PATH|Path to the downloaded YOLOv5 repository|C:/yolov5|
|MODELS_PATH|Path to the folder containing the model(s) to be used|C:/models|

**2. Usage of the script**
Arguments for the script:
|Argument Name|Description|Default Value|
|:-:|-|:-:|
|w|choose webcam as source of detection|-|
|v|choose a video as source of detection|-|
|i|choose an image as source of detection|-|
|model|pretrained model to be used for detection|model
|threshold|threshold for the detection|0.5|

Example commands:

    person_detector.py -w --model "12b100e" --threshold "0.3" # webcam with 0.3 threshold and 12b100e.pt model weights
    person_detector.py -i "image.jpg" # use image.jpg as source with 0.5 threshold and model.pt model weights
    person_detector.py -v "video.mp4" # use video.mp4 as source with 0.5 threshold and model.pt model weights


## ImageDownloader
Python script that allows downloading images with "Pexels API".
Documentation: https://www.pexels.com/api/documentation
