from donkeycar.gym import remote_controller
import numpy as np
import time

class DonkeySim:
    
    def __init__(self, car = "kari", server = "mqtt.eclipse.org"):
        self.control = remote_controller.DonkeyRemoteContoller(car, server)
        self.state = self.control.observe()

        self.throttle = 0

    def reset(self):
        self.control.take_action(action=[0, 0])
        self.throttle = 0
        time.sleep(1)
        self.state = self.control.observe()
        return self.state
    
    def step(self, control, step_length):
        
        #steering = control[0]
        #self.throttle = max(min(1, self.throttle + control[1]), 0.1)

        self.control.take_action(action=control)
        time.sleep(step_length)
        obs = self.control.observe()
        prev_state = self.state
        self.state = obs
        done = self.is_dead(prev_state)

        return control, self.state, done

    def is_dead(self, prev_state):

        crop_height = 40
        required = 0.8
        
        cropped = self.state[-crop_height:]

        rgb = cropped[:,:,0] > cropped[:,:,2]

        is_stopped = np.isclose(self.state, prev_state).sum() > 30000

        return rgb.sum() / (crop_height * 160) > required# or is_stopped


       #darkness = len(self.state[(self.state > 120) * (self.state < 130)])

        #if darkness < 2300:
        #    return 1.0
        #else:
        #    return 0.0
        