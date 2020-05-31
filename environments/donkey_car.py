from donkeycar.gym import remote_controller
import numpy as np
import time

class DonkeyCar:
    
    def __init__(self, car = "kari", server = "mqtt.eclipse.org"):
        self.control = remote_controller.DonkeyRemoteContoller(car, server)
        self.state = self.control.observe()
        self.action = [0, 0]
        self.throttle = 0.15

    def reset(self):
        self.control.take_action(action=[0, 0])
        time.sleep(1)
        self.state = self.control.observe()

        throttle_input = input("Input throttle. Previous was {}\n".format(self.throttle))

        if throttle_input:
            try:
                self.throttle = float(throttle_input)
            except:
                pass

        return self.state
    
    def step(self, control):
        
        control[1] = self.throttle
        self.control.take_action(action=control)

        time.sleep(0.1)
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