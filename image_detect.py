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
            # #cv2.putText(frame, label + "%", (x1-40, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # # cv2.putText(frame, f"Total Targets: {numDetect}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


def run_detect():
    script_path = os.path.dirname(os.path.realpath(__file__))

    img_name = "20240201101106"
    img_folder = os.path.abspath(os.path.join(script_path, "images"))

    trained_model_folder = os.path.abspath(os.path.join(script_path, "weights"))
    trained_model = os.path.join(trained_model_folder, "ATRIS_unistrut20240221v8.pt")
    model = YOLO(trained_model)

    # OpenCV function imread to load image
    image_file = os.path.join(img_folder, img_name)
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

    #detect = model.predict(source=frameL, save=True, save_txt=True)
    detect = model.predict(source=image)
    frame= plot_boxes(model, detect, image) 

    cv2.imshow("Left", frame)
    c = cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_detect()

