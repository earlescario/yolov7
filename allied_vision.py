import time
from pathlib import Path
import sys # For Allied Vision
from typing import Optional # For Allied Vision
from queue import Queue # For Allied Vision
from types import SimpleNamespace # For hardcoding 'opt'

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np # For letterbox and other operations

# --- YOLOv7 Imports ---
from models.experimental import attempt_load
# from utils.datasets import letterbox # We will copy this function
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, Detections
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

# --- Allied Vision VmbPy Imports ---
from vmbpy import VmbSystem, Camera, Stream, Frame, FrameStatus, PixelFormat, VmbCameraError, VmbFeatureError, \
                  intersect_pixel_formats, COLOR_PIXEL_FORMATS, MONO_PIXEL_FORMATS


# --- Helper function from YOLOv7 utils.datasets (or utils.augmentations) ---
# This is often needed for resizing images while maintaining aspect ratio
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


# --- Allied Vision Camera Setup Code (from asynchronous_grab_opencv.py) ---
opencv_display_format = PixelFormat.Bgr8 # All frames will be converted to this

def print_preamble_av(): # Renamed to avoid conflict
    print('///////////////////////////////////////////////////')
    print('/// VmbPy Asynchronous Grab with OpenCV Example ///')
    print('///////////////////////////////////////////////////\n')

def abort_av(reason: str, return_code: int = 1): # Renamed
    print(reason + '\n')
    sys.exit(return_code)

def get_camera_av(camera_id: Optional[str]) -> Camera: # Renamed
    with VmbSystem.get_instance() as vmb:
        if camera_id:
            try:
                return vmb.get_camera_by_id(camera_id)
            except VmbCameraError:
                abort_av('Failed to access Camera \'{}\'. Abort.'.format(camera_id))
        else:
            cams = vmb.get_all_cameras()
            if not cams:
                abort_av('No Cameras accessible. Abort.')
            return cams[0]

def setup_camera_av(cam: Camera): # Renamed
    with cam:
        try:
            cam.ExposureAuto.set('Continuous')
        except (AttributeError, VmbFeatureError): pass
        try:
            cam.BalanceWhiteAuto.set('Continuous')
        except (AttributeError, VmbFeatureError): pass
        try:
            stream = cam.get_streams()[0]
            stream.GVSPAdjustPacketSize.run()
            while not stream.GVSPAdjustPacketSize.is_done(): pass
        except (AttributeError, VmbFeatureError): pass

def setup_pixel_format_av(cam: Camera): # Renamed
    cam_formats = cam.get_pixel_formats()
    cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
    convertible_color_formats = tuple(f for f in cam_color_formats
                                      if opencv_display_format in f.get_convertible_formats())
    cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
    convertible_mono_formats = tuple(f for f in cam_mono_formats
                                     if opencv_display_format in f.get_convertible_formats())

    if opencv_display_format in cam_formats:
        cam.set_pixel_format(opencv_display_format)
    elif convertible_color_formats:
        cam.set_pixel_format(convertible_color_formats[0])
    elif convertible_mono_formats:
        cam.set_pixel_format(convertible_mono_formats[0])
    else:
        abort_av('Camera does not support an OpenCV compatible format. Abort.')

class HandlerAV: # Renamed
    def __init__(self):
        self.display_queue = Queue(maxsize=10) # Use maxsize for clarity

    def get_image(self, timeout=None): # Added timeout
        try:
            return self.display_queue.get(timeout=timeout)
        except Queue.Empty:
            return None


    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            if frame.get_pixel_format() == opencv_display_format:
                display_frame_np = frame.as_opencv_image()
            else:
                # This creates a copy. Original `frame` can be requeued.
                converted_frame = frame.convert_pixel_format(opencv_display_format)
                display_frame_np = converted_frame.as_opencv_image()
            
            try:
                self.display_queue.put_nowait(display_frame_np)
            except Queue.Full:
                # print("Warning: Display queue full, dropping frame.")
                pass # Or handle frame drop

        cam.queue_frame(frame)

# --- Main Detection Logic ---
def run_detection():
    # --- Hardcoded YOLOv7 Parameters (equivalent to opt object) ---
    opt = SimpleNamespace()
    opt.weights = 'models/EDA2/best.pt' # IMPORTANT: Path to your model
    # opt.weights = 'yolov7.pt' # Or a standard YOLOv7 model
    opt.img_size = 640
    opt.conf_thres = 0.5
    opt.iou_thres = 0.45
    opt.device = ''  # Default: auto-select (cuda if available, else cpu)
    # opt.device = 'cpu' # To force CPU
    opt.view_img = True # We will display using OpenCV
    opt.save_txt = False # Set to True if you want to save labels
    opt.save_conf = False
    opt.nosave = False # If True, will not save output video
    opt.classes = None # Filter by class: --class 0, or --class 0 2 3
    opt.agnostic_nms = False
    opt.augment = False
    opt.project = 'runs/detect_live'
    opt.name = 'exp_allied_vision'
    opt.exist_ok = False # if True, existing project/name ok, do not increment
    opt.no_trace = True # Set to False if you want to trace the model

    # --- YOLOv7 Initialization ---
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size

    if not opt.no_trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference once for warmup
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    # --- Allied Vision Camera Initialization ---
    print_preamble_av()
    # cam_id = None # Use first available camera
    cam_id = None # Or specify your camera ID string here if you have multiple
    
    # Output video writer
    save_vid_path = None
    vid_writer = None
    if not opt.nosave:
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        save_dir.mkdir(parents=True, exist_ok=True)
        # Note: FPS and frame size will be set once first frame is received
        # For now, define the path. We'll initialize VideoWriter later.
        save_vid_path = str(save_dir / "live_output.mp4")


    with VmbSystem.get_instance():
        with get_camera_av(cam_id) as cam:
            print(f"Using camera: {cam.get_name()}")
            setup_camera_av(cam)
            setup_pixel_format_av(cam)
            handler = HandlerAV()

            try:
                cam.start_streaming(handler=handler, buffer_count=10)
                print(f"Streaming from '{cam.get_name()}'. Press 'q' or <Esc> to stop.")

                t0 = time.time()
                frames_processed = 0

                while True:
                    im0 = handler.get_image(timeout=0.1) # Get frame from camera queue (OpenCV BGR format)
                    
                    if im0 is None: # If no frame, continue or wait
                        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]: # Check for quit key even if no frame
                            break
                        continue
                    
                    # Padded resize
                    img = letterbox(im0, imgsz, stride=stride)[0]

                    # Convert BGR to RGB, HWC to CHW
                    img = img[:, :, ::-1].transpose(2, 0, 1)
                    img = np.ascontiguousarray(img)

                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=opt.augment)[0]
                    t2 = time_synchronized()

                    # Apply NMS
                    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                    t3 = time_synchronized()

                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        im0c = im0.copy() # Create a copy of the original frame for drawing
                        gn = torch.tensor(im0c.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0c.shape).round()

                            # Print results
                            s = ""
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if opt.view_img or not opt.nosave:  # Add bbox to image
                                    label = f'{names[int(cls)]} {conf:.2f}'
                                    plot_one_box(xyxy, im0c, label=label, color=colors[int(cls)], line_thickness=2) # Thicker line for visibility

                                # if opt.save_txt: # Example of saving labels
                                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                #     with open(txt_path + '.txt', 'a') as f: # Define txt_path if using
                                #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            
                            print(f'{s}Done. ({(t2 - t1)*1000:.1f}ms) Inference, ({(t3 - t2)*1000:.1f}ms) NMS')


                        # Display the Detections
                        if opt.view_img:
                            cv2.imshow(f"YOLOv7 Live - {cam.get_name()}", im0c)
                            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:  # 1 millisecond, check for 'q' or ESC
                                raise KeyboardInterrupt # To break the outer loop cleanly

                        # Save video
                        if not opt.nosave and save_vid_path:
                            if vid_writer is None: # Initialize video writer with first frame info
                                fps_vid = 20 # Desired output FPS, camera might be faster
                                w, h = im0c.shape[1], im0c.shape[0]
                                vid_writer = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_vid, (w, h))
                            vid_writer.write(im0c)
                        
                        frames_processed += 1

            except KeyboardInterrupt:
                print("\nStream stopped by user.")
            finally:
                cam.stop_streaming()
                if vid_writer:
                    vid_writer.release()
                    print(f"Saved output video to {save_vid_path}")
                cv2.destroyAllWindows()
                print(f"Total frames processed: {frames_processed}")
                print(f"Total time: {time.time() - t0:.3f}s")
                if frames_processed > 0:
                     print(f"Average FPS: {frames_processed / (time.time() - t0):.2f}")


if __name__ == '__main__':
    # Ensure this script is run from the yolov7 root directory for imports to work,
    # or adjust PYTHONPATH.
    # Example: python path/to/this/live_detect_allied_vision.py
    with torch.no_grad():
        run_detection()