import os
import time
import webbrowser
import argparse
from tensorboard import program

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yoloversion', type=int, default=5, help='YOLO version to be used')
    parser.add_argument('--resultdir', type=str, default='C:\YOLOtraining', help='directory for the output of the model trainings')
    return parser.parse_args()


def show_tensorboard(result_dir, yolo_version):
    tb = program.TensorBoard()
    if not os.path.isdir(result_dir):
        print(f"Result directory [{result_dir}] does not exist.")
        exit()
    os.chdir(result_dir)
    if yolo_version == 5:
        tb.configure(logdir='runs/train')
    elif yolo_version == 8:
        tb.configure(logdir='runs/detect')
    else:
        print('YOLO version shall be 5 or 8.')
        exit()
    url = tb.launch()
    chrome_path = os.getenv('CHROME_PATH', 'C:\\Program Files\\Google\\Chrome\\Application')
    webbrowser.register('chrome',None,webbrowser.BackgroundBrowser(f"{chrome_path}\\chrome.exe"))
    webbrowser.get('chrome').open(url)
    print(f"To stop the tensorboard server type 'taskkill /F /PID {os.getpid()}'")
    while True:
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            break
    print()
    print("Shutting down")


if __name__ == '__main__':
    opt=parse_opt()
    show_tensorboard(opt.resultdir, opt.yoloversion)
