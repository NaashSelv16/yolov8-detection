import cv2
import torch
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

def plot_boxes(model, detect, frame):
    """
    plots boxes and labels on frame.
    :param results: inferences made by model
    :param frame: frame on which to  make the plots
    :return: new frame with boxes and labels plotted.
    """
    for d in detect:
        annotator = Annotator(frame)
        
        boxes = d.boxes
        for box in boxes:
            
            b = box.xyxy.tolist()[0]  # get box coordinates in (left, top, right, bottom) format
            x1,y1,x2,y2 = int(b[0]),int(b[1]),int(b[2]),int(b[3])
            
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
            classes = model.names
            bgr = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

            object_index  = int(box.cls.tolist()[0])
            label = str(classes[object_index])

            # cv2.putText(frame, label, (x1-40, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #cv2.putText(frame, label + "%", (x1-40, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # # cv2.putText(frame, f"Total Targets: {numDetect}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


def run_camera():
    script_path = os.path.dirname(os.path.realpath(__file__))

    img_folder = os.path.abspath(os.path.join(script_path, "images"))

    trained_model_folder = os.path.abspath(os.path.join(script_path, "weights"))
    trained_model = os.path.join(trained_model_folder, "Nissan_defect20240828.pt")
    model = YOLO(trained_model)

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()

        #detect = model.predict(source=frameL, save=True, save_txt=True)
        detect = model.predict(source=frame)
        frame= plot_boxes(model, detect, frame)

        #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_camera()

