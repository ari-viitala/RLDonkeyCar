from donkeycar.gym import remote_controller
import numpy as np
import time

class DonkeyCar:
    
    def __init__(self, car = "kari", server = "mqtt.eclipse.org"):
        self.control = remote_controller.DonkeyRemoteContoller(car, server)
        self.state = self.control.observe()

    def reset(self):
        self.control.take_action(action=[0, 0])
        time.sleep(1)
        self.state = self.control.observe()
        return self.state
    
    def step(self, control):
        
        self.control.take_action(action=control)
        time.sleep(0.1)
        obs = self.control.observe()
        self.state = obs
        return self.state

    def is_dead(self):
        darkness = len(im[(im > 120) * (im < 130)])

        if darkness < threshold:
            return True
        else:
            return False
        