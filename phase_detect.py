import cv2
import torch
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import phase.pyphase as phase

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
    # Define information about the Phase camera
    # Each camera has unique camera_name, left_serial, and right_serial
    # camera_name = "746974616e24318"
    # left_serial = "40091829"
    # right_serial = "40098273"
    # device_type = phase.stereocamera.CameraDeviceType.DEVICE_TYPE_TITANIA
    # interface_type = phase.stereocamera.CameraInterfaceType.INTERFACE_TYPE_USB
    camera_name = "70686f626f24316"
    left_serial = "22864912"
    right_serial = "24512570"
    device_type = phase.stereocamera.CameraDeviceType.DEVICE_TYPE_PHOBOS
    interface_type = phase.stereocamera.CameraInterfaceType.INTERFACE_TYPE_GIGE

    # Define parameters for read process
    downsample_factor = 1.0
    display_downsample = 0.25
    exposure_value = 50000

    script_path = os.path.dirname(os.path.realpath(__file__))

    trained_model_folder = os.path.abspath(os.path.join(script_path, "weights"))
    trained_model = os.path.join(trained_model_folder, "Nissan_defect20240828.pt")
    model = YOLO(trained_model)

    # Create stereo camera device information from parameters
    device_info = phase.stereocamera.CameraDeviceInfo(
        left_serial, right_serial, camera_name,
        device_type,
        interface_type)
    # Create stereo camera
    # phaseCam = phase.stereocamera.TitaniaStereoCamera(device_info)
    phaseCam = phase.stereocamera.PhobosStereoCamera(device_info)
    ret = phaseCam.connect()
    phaseCam.enableHardwareTrigger(False)

    if (ret):
        phaseCam.startCapture()
        # Set camera exposure value
        phaseCam.setExposure(exposure_value)
        print("Running camera capture...")
        while phaseCam.isConnected():
            read_result = phaseCam.read()

            frame = read_result.left

            detect = model.predict(source=frame)
            frame= plot_boxes(model, detect, frame)

            # Display stereo and disparity images
            img_left = phase.scaleImage(
                    frame, display_downsample)

            #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.imshow('Input', img_left)

            c = cv2.waitKey(1)
            if c == 27:
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_camera()

