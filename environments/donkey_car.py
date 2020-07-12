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
    
    def step(self, control, step_length):
        
        steering = control[0]
        throttle = self.throttle if control[1]> 0 else 0
        action = [steering, throttle]
        self.control.take_action(action=action)

        time.sleep(step_length)
        obs = self.control.observe()
        self.state = obs
        done = self.is_dead()

        return [steering, 0], self.state, done

    def is_dead(self):

        crop_height = 20
        crop_width = 20
        threshold = 70
        pixels_percentage = 0.10

        pixels_required = (self.state.shape[1] - 2 * crop_width) * crop_height * pixels_percentage

        #(im[:,:,0] > threshold) & (im[:,:,1] < 150) & (im[:,:,1] < 150)

        crop = self.state[-crop_height:, crop_width:-crop_width]
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
