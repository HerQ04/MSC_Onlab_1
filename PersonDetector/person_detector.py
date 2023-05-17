import argparse
import cv2
import torch
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', action='store_true', help='the flag for webcam usage')
    parser.add_argument('-v', type=str, help='the name of the video file')
    parser.add_argument('-i', type=str, help='the name of the image file')
    parser.add_argument('--model', type=str, default='model', help='the name of the pretrained model file (without the file extension)')
    parser.add_argument('--threshold', type=float, default=0.5, help='the threshold to be used during detection')
    return parser.parse_args()


def detectx (frame, model):
    results = model([frame])
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates


def plot_boxes(results, frame, classes):
    labels, cord = results
    n = len(labels) #number of detections
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= threshold: ### threshold value for detection. We are discarding everything below this value
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]
            if text_d == 'person':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

    return frame


def perform_detection(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detectx(frame, model)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = plot_boxes(results, frame, classes = model.names)
    return frame


def video(model, video_path):
    print(f"processing video: {video_path}...")
    cap = cv2.VideoCapture(video_path)
    frame_no = 1
    window_name = f"{video_path} -> detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if ret :
            frame = perform_detection(frame, model)
            cv2.imshow(window_name, frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break
        frame_no += 1
    
    cv2.destroyAllWindows()
    

def image(model, image_path):
    print(f"processing image: {image_path}...")
    frame = cv2.imread(image_path)
    frame = perform_detection(frame, model)
    window_name = f"{image_path} -> detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break


def webcam(model):
    cap = cv2.VideoCapture(0)
    window_name = 'Webcam -> detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_no = 1
    while True:
        ret, frame = cap.read()
        if ret :
            frame = perform_detection(frame, model)
            cv2.imshow(window_name, frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        frame_no += 1
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = parse_opt()
    if(not opt.w and opt.v == None and opt.i == None):
        print('Please choose a usage type. -w for webcam, -v "videoPath.mp4" for video, or -i "imagePath.jpg" for image.')
        exit()
    
    if((opt.w and opt.v != None) or (opt.w and opt.i != None) or (opt.i != None and opt.v != None)):
        print('Please only select one source for detection. -w for webcam, -v "videoPath.mp4" for video, or -i "imagePath.jpg" for image.')
        exit()
    
    modelsPath = os.getenv('MODELS_PATH', 'C:/models')
    if(not os.path.exists(f"{modelsPath}/{opt.model}.pt")):
        print(f"Model file doesn't exist: '{modelsPath}/{opt.model}.pt'.")
        exit()

    yolov5Path = os.getenv('YOLOV5_PATH', 'C:/yolov5')
    
    model = torch.hub.load(yolov5Path, 'custom', source ='local', path=f"{modelsPath}/{opt.model}.pt", force_reload=True)

    global threshold
    threshold = opt.threshold
    
    if(opt.w):
        webcam(model)
    elif(opt.v != None):
        video(model, opt.v)
    elif(opt.i != None):
        image(model, opt.i)
