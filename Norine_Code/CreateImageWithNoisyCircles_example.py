runfile('C:\\Users\\eqp83935\\workspace\\data\\src\\simple_circle_detector.py')
import numpy as np
image = np.zeros((1000,1000))
draw_circle(image, 400, 300, 200, 4)
draw_circle(image, 150, 750, 50, 2)
draw_circle(image, 700, 700, 150, 2)
draw_circle(image, 850, 150, 75, 5)
image_saving = image
image = add_noise(image, 0.1)