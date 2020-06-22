from donkeycar.gym import remote_controller
import numpy as np
import time

class DonkeySim:
    
    def __init__(self, car = "kari", server = "mqtt.eclipse.org"):
        self.control = remote_controller.DonkeyRemoteContoller(car, server)
        self.state = self.control.observe()

    def reset(self):
        self.control.take_action(action=[0, 0])
        time.sleep(1)
        self.state = self.control.observe()
        return self.state
    
    def step(self, control, step_length):
        
        self.control.take_action(action=control)
        time.sleep(step_length)
        obs = self.control.observe()
        self.state = obs
        done = self.is_dead()

        return control, self.state, done

    def is_dead(self):
        darkness = len(self.state[(self.state > 120) * (self.state < 130)])

        if darkness < 2500:
            return 1.0
        else:
            return 0.0
        