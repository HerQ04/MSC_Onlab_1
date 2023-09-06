import torch
import yaml
import os
import sys
import argparse
import webbrowser
import time
import results
from projectsetup import setup_project
from tensorboard import program

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov8', action='store_true', help='the flag for using YOLOv8')
    parser.add_argument('--img', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--batch', type=int, default=-1, help='total batch size for all GPUs, use -1 for autobatch')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--model', type=str, default='yolov5m', help='model to be used for training')
    parser.add_argument('--name', default='exp', help='name of the current project output')
    parser.add_argument('--dataset', default='person-4', help='name of the dataset')
    parser.add_argument('--outputdir', type=str, default='C:/yolov5', help='the folder where the result of the training inside the runs/train folder shall be saved')
    parser.add_argument('--workspace', type=str, default='msc-onlab-1', help='the Roboflow workspace containing the project')
    parser.add_argument('--project', type=str, default='person-so5ko', help='the id of the Roboflow project')
    parser.add_argument('--version', type=int, default=4, help='the version of the dataset')
    return parser.parse_args()


def setup_training(opt):
    #Print used device
    print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
    
    os.chdir(opt.outputdir)
    global dataset_location
    dataset_location = f"{opt.outputdir}/{opt.dataset}"
    #Setup Roboflow project if it doesn't exist
    if not os.path.isdir(opt.dataset):
        dataset_location = setup_project(opt.workspace, opt.project, opt.version, 8 if opt.yolov8 else 5)
    
    #Print Roboflow dataset info
    with open(f"{dataset_location}/data.yaml") as f:
        print(f.read())


def perform_training(opt):
    if opt.yolov8:
        #Train model with YOLOv8
        model = YOLO(f"{opt.model}.pt")
        model.train(imgsz=opt.img, batch=opt.batch, epochs=opt.epochs, data=f"{dataset_location}/data.yaml", name=opt.name, patience=0)
    else:
        #Train model with YOLOv5
        train.run(img=opt.img, batch=opt.batch, epochs=opt.epochs, data=f"{dataset_location}/data.yaml", cfg=f"{yolov5path}/models/{opt.model}.yaml", weights=f"{opt.model}.pt", name=opt.name)



if __name__ == '__main__':
    opt = parse_opt()
    global yolov5path
    yolov5path = os.getenv('YOLOV5_PATH', 'C:/yolov5')
    if opt.yolov8:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError:
            raise ModuleNotFoundError("""Failed to import 'YOLO' from module 'ultralytics'. Maybe you forgot to install YOLOv8 using the following command: 'pip install ultralytics'.""")
    else:
        sys.path.append(yolov5path)
        try:
            import train
        except ModuleNotFoundError:
            raise ModuleNotFoundError("""Failed to import module 'train'. If your YOLOv5 is not located at 'C:\yolov5' maybe you forgot to set the YOLOV5_PATH system environment variable.""")
    
    setup_training(opt)
    perform_training(opt)
    //results.show_tensorboard(opt.outputdir, 8 if opt.yolov8 else 5)
