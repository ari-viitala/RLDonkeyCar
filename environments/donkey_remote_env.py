from donkeycar.parts.network import MQTTValueSub, MQTTValuePub
from donkeycar.parts.image import JpgToImgArr

import numpy as np
import time

class DonkeyRemoteEnv:
    
    def __init__(self, car_name="Kari", mqtt_broker="mqtt.eclipse.org", realsense=False, env_type="DonkeyCar"):

        self.camera_sub = MQTTValueSub("donkey/%s/camera" % car_name, broker=mqtt_broker)

        if realsense:
            self.state_sub = MQTTValueSub("donkey/%s/state" % car_name, broker=mqtt_broker)

        self.controller_pub = MQTTValuePub("donkey/%s/controls" % car_name, broker=mqtt_broker)

        self.jpgToImg = JpgToImgArr()

        self.img = np.zeros((40, 40))
        self.state = None
        self.speed = 0
        self.realsense = realsense
        self.sim = env_type != "DonkeyCar"
        

    def observe(self):

        self.img = self.jpgToImg.run(self.camera_sub.run())

        if self.realsense:
            self.state = self.state_sub.run()
            v = self.state["v"]
            self.speed = (v[0]**2 + v[1]**2 + v[2]**2)**0.5

    def reset(self):
        self.controller_pub.run([0, 0])

        self.observe()
        time.sleep(2)

        return self.img
    
    def step(self, control, step_length):
        print(control)
        self.controller_pub.run(control)
        time.sleep(step_length)

        self.observe()

        done = self.is_dead_sim() if self.sim else self.is_dead_real()

        return self.img, done

    def is_dead_real(self):

        crop_height = 20
        crop_width = 20
        threshold = 70
        pixels_percentage = 0.10

        pixels_required = (self.img.shape[1] - 2 * crop_width) * crop_height * pixels_percentage

        crop = self.img[-crop_height:, crop_width:-crop_width]

        r = crop[:,:,0] < threshold
        g = crop[:,:,1] < threshold
        b = crop[:,:,2] < threshold

        pixels = (r & g & b).sum()

        print("Pixels: {}, Required: {}".format(pixels, pixels_required))
        
        return  pixels < pixels_required

    def is_dead_sim(self):

        crop_height = 40
        required = 0.8
        
        cropped = self.img[-crop_height:]

        rgb = cropped[:,:,0] > cropped[:,:,2]

        return rgb.sum() / (crop_height * 160) > required