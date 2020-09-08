from donkeycar.parts.network import MQTTValueSub, MQTTValuePub
from donkeycar.parts.image import JpgToImgArr

import numpy as np
import time

class DonkeyCarSpeed:
    
    def __init__(self, car="kari", mqtt_broker="mqtt.eclipse.org"):

        self.camera_sub = MQTTValueSub("donkey/%s/camera" % car, broker=mqtt_broker)
        self.state_sub = MQTTValueSub("donkey/%s/state" % car, broker=mqtt_broker)

        self.controller_pub = MQTTValuePub("donkey/%s/controls" % car, broker=mqtt_broker)
        self.jpgToImg = JpgToImgArr()

        self.img = np.zeros((40, 40))
        self.state = None
        self.speed = 0
        

    def observe(self):

        self.img = self.jpgToImg.run(self.camera_sub.run())
        self.state = self.state_sub.run()
  
    def reset(self):
        self.controller_pub.run([0, 0])

        self.observe()
        time.sleep(1)

        return self.img
    
    def step(self, control, step_length):
        
        self.controller_pub.run(control)
        time.sleep(step_length)

        self.observe()

        v = self.state["v"]
        self.speed = (v[0]**2 + v[1]**2 + v[2]**2)**0.5

        done = self.is_dead()

        return self.img, done

    def is_dead(self):

        crop_height = 20
        crop_width = 20
        threshold = 70
        pixels_percentage = 0.10

        pixels_required = (self.img.shape[1] - 2 * crop_width) * crop_height * pixels_percentage

        #(im[:,:,0] > threshold) & (im[:,:,1] < 150) & (im[:,:,1] < 150)

        crop = self.img[-crop_height:, crop_width:-crop_width]
        #gs = np.dot(crop, [0.299, 0.587, 0.114])

        r = crop[:,:,0] < threshold
        g = crop[:,:,1] < threshold
        b = crop[:,:,2] < threshold

        pixels = (r & g & b).sum()
        
        #im = self.state[-crop_height:, crop_width:-crop_width]
        #gs = (im[:,:,0] > 150) & (im[:,:,1] < 150) & (im[:,:,1] < 150)

        #pixels = len(gs[gs])

        #pixels = len(gs[gs < threshold])

        print("Pixels: {}, Required: {}".format(pixels, pixels_required))
        
        return  pixels < pixels_required
