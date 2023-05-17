import argparse
from roboflow import Roboflow

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yoloversion', type=int, default=5, help='YOLO version to be used')
    parser.add_argument('--workspace', type=str, default='msc-onlab-1', help='the Roboflow workspace containing the project')
    parser.add_argument('--project', type=str, default='person-so5ko', help='the id of the Roboflow project')
    parser.add_argument('--version', type=int, default=4, help='the version of the dataset')
    return parser.parse_args()


def setup_project(workspace="msc-onlab-1", project="person-so5ko", version=4, yoloversion=5):
    rf = Roboflow(api_key="FYXm4j188EfCcP2ODqYO")
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download(f"yolov{yoloversion}")
    return dataset.location


if __name__ == '__main__':
    opt = parse_opt()
    ds_location = setup_project(opt.workspace, opt.project, opt.version, opt.yoloversion)
    print(f"Roboflow project dataset downloaded to {ds_location}")
