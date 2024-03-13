#import sys
#import numpy as np
import cv2
import time
from time import sleep
from datetime import timezone
from azupload import *

import os
import numpy as np
try:
  from queue import Queue
except ImportError:
  from Queue import Queue
import platform
import picamera2
import logging

logging.basicConfig(level=logging.DEBUG)

ptuvc_path='/home/pi/sources/purethermal1-uvc-capture/python/'
logging.info(ptuvc_path)
sys.path.insert(0,ptuvc_path)
from uvctypes import *

from picamera2 import Picamera2
from libcamera import controls

import logging 

#image resolution
ry,rx=1920,1440#2592,1952#3840,2160#160,128#256,192#960,544#1280,960#
exp_time = 1000    
frame_time = 500000//30
brightness = 0.0
contrast = 1.0

home = os.path.join("/home","pi")
p = os.path.join(home,'CookIRCamET','Images')
web = '/var/www/html'#os.path.join(home,'CookIRCamET','static')

BUF_SIZE = 2
q = Queue(BUF_SIZE)

def py_frame_callback(frame, userptr):
  array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
  data = np.frombuffer(
    array_pointer.contents, dtype=np.dtype(np.uint16)
  ).reshape(
    frame.contents.height, frame.contents.width
  ) # no copy
  
  if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
    return

  if not q.full():
    q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

class ir_cam:
  def __init__(self,waittime):
    self.ctx = POINTER(uvc_context)()
    self.ctrl = uvc_stream_ctrl()
    self.waittime=waittime
    res = libuvc.uvc_init(byref(self.ctx), 0)
    if res < 0:
      print("uvc_init error")
      exit(1)
    self.get_devices()
      
    if len(self.devs) == 0:
      print("Did not find any devices")
      exit(1)

    print("Found {} devices".format(len(self.devs)))

        
    second = False
    for (desc, self.dev) in self.devs:
      self.devh = POINTER(uvc_device_handle)()
      res = libuvc.uvc_open(self.dev, byref(self.devh))
      if res == 0:
        break
      print("could not open {}, trying next".format(desc.serialNumber))
      second = True

    if res < 0:
      print("Could not open any devices")
      exit(1)

    print("device opened: ", desc.manufacturer, desc.product, desc.serialNumber)

    print_device_info(self.devh)
    print_device_formats(self.devh)

    frame_formats = uvc_get_frame_formats_by_guid(self.devh, VS_FMT_GUID_Y16)
    if len(frame_formats) == 0:
      print("device does not support Y16")
      exit(1)

    libuvc.uvc_get_stream_ctrl_format_size(self.devh, byref(self.ctrl), UVC_FRAME_FORMAT_Y16,
                                           frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
                                           )

    res = libuvc.uvc_start_streaming(self.devh, byref(self.ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
    if res < 0:
      print("uvc_start_streaming failed: {0}".format(res))
      exit(1)
    return None
      
  def get_devices(self):

    self.devs = []

    devs = POINTER(c_void_p)()
    res = libuvc.uvc_get_device_list(self.ctx, byref(devs))

    if res != 0:
      print("uvc_find_device error")
      exit(1)

    count = 0
    while devs[count] != None:
      dev = cast(devs[count], POINTER(uvc_device))
      count += 1

      desc = POINTER(uvc_device_descriptor)()
      res = libuvc.uvc_get_device_descriptor(dev, byref(desc))

      if res != 0:
        print("Could not get device descriptor")
        continue

      if desc.contents.idProduct == PT_USB_PID and desc.contents.idVendor == PT_USB_VID:
        self.devs.append((desc.contents, dev))

  def capture(self):
    now = datetime.now(timezone.utc)
    current_time = now.strftime("%Y%m%d_%H%M%S")
    ir = q.get(True, 500)            
    fname = current_time+'_ir.png'
    logging.info(os.path.join(p,fname))
    cv2.imwrite(os.path.join(p,fname),ir)
    upload_file(raw,os.path.join(p,fname),'./CookIRCamET/Images/CookHY2024/V3/'+fname)

    cv2.normalize(ir, ir, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(ir, 8, ir)
    cv2.imwrite(os.path.join(web,'bar.bmp'),np.uint8(ir))
    logging.info(os.path.join(web,'bar.bmp'))
    return ir

  def run(self):
    while True:
      self.capture()
      sleep(self.waittime)
  
  def __del__(self):
    libuvc.uvc_stop_streaming(self.devh)
    libuvc.uvc_unref_device(self.dev)
    libuvc.uvc_exit(self.ctx)

class bgr_cam:
  def __init__(self,rx,ry,exp_time,frame_time,brightness,contrast,waittime):
    self.waittime=waittime
    self.cam_bgr = Picamera2(0)
    camera_config = self.cam_bgr.create_still_configuration(main={"size": (rx, ry)})
    self.cam_bgr.configure(camera_config)
    self.cam_bgr.controls.ExposureTime = exp_time
    self.cam_bgr.controls.AwbEnable = False
    self.cam_bgr.controls.AeEnable = False
    self.cam_bgr.controls.NoiseReductionMode = controls.draft.NoiseReductionModeEnum.Off   
#    self.cam_bgr.set_controls({"AfMode":controls.AfModeEnum.Continuous,
#                               "Brightness":brightness,
#                               "Contrast":contrast,             
#                               "FrameDurationLimits": (frame_time, frame_time)})
    self.cam_bgr.set_controls({"AfMode":controls.AfModeEnum.Auto,
                               "Brightness":brightness,
                               "Contrast":contrast,             
                               "FrameDurationLimits": (frame_time, frame_time)})

    logging.info(self.cam_bgr.controls)
    self.cam_bgr.start()
    success = self.cam_bgr.autofocus_cycle()
    return None

  def capture(self):
    now = datetime.now(timezone.utc)
    current_time = now.strftime("%Y%m%d_%H%M%S")
    r = self.cam_bgr.capture_array()
    r = np.flip(r,axis=2)
    fname = current_time+'_bgr.png'
    logging.info(os.path.join(p,fname))
    cv2.imwrite(os.path.join(p,fname),r)
    upload_file(raw,os.path.join(p,fname),'./CookIRCamET/Images/CookHY2024/V3/'+fname)
    logging.info(os.path.join(web,'foo.bmp'))
    cv2.imwrite(os.path.join(web,'foo.bmp'),r)
    return r

  def run(self):
    while True:
      self.capture()
      sleep(self.waittime)
  
  def __del__(self):
      self.cam_bgr.stop()
