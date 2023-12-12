import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pickle

sensornames = ['color', 'icub_left', 'icub_right']
toolnames = ['hook', 'ruler', 'spatula', 'sshot']
actions = ['left_to_right', 'pull', 'push', 'right_to_left']
objectnames = ['0_woodenCube', '1_pearToy', '2_yogurtYellowbottle', '3_cowToy', '4_tennisBallYellowGreen',
               '5_blackCoinBag', '6_lemonSodaCan', '7_peperoneGreenToy', '8_boxEgg', '9_pumpkinToy',
               '10_tomatoCan', '11_boxMilk', '12_containerNuts', '13_cornCob', '14_yellowFruitToy',
               '15_bottleNailPolisher', '16_boxRealSense', '17_clampOrange', '18_greenRectangleToy', '19_ketchupToy']
width, height = 128,128

### FUNCTIONS ###
def concatenate(image_1, image_2):
    # Concatenate the image
    return np.concatenate((image_1, image_2), axis=1)

def normalize(image):
    # Normalize the image
    return image/255

def resize(image, width, height):
    # Resize an image to a fixed size
    return cv2.resize(image, (width, height))

def bgr_to_rgb(image):
    # Changes the images from BGR to RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def image_from_array(image):
    # Changes the array to image
    return Image.fromarray(image)

def flatten(image, width, height):
    # Flatten the images
    return image.flatten().reshape(1, width*height*3)

def to_float(image):
    # Convert to float32
    return image.astype(np.float32)
def main():
    for a in range(len(objectnames)):
        objectname = objectnames[a]
        for y in range(len(toolnames)):
            toolname = toolnames[y]
            for x in range(len(actions)):
                action = actions[x]
                label = actions.index(action)
                
                # Split into sets: 60% Training, 20% Validation, 20% Testing
                ids = np.random.choice(np.arange(10), 10, replace=False)
                training_ids, validation_ids, testing_ids = ids[0:6], ids[6:8], ids[8:10]
                # Loop through the number of repeats
                for j in range(10):
                    if sensor == 'icub_right' or sensor == 'icub_left':
                        init = cv2.imread(path + 'init_color_' + sensor + '_' + str(j) + '.png')
                        effect = cv2.imread(path + 'effect_color_' + sensor + '_' + str(j) + '.png')
                    else:
                        init = cv2.imread(path + 'init_' + sensor + '_' + str(j) + '.png')
                        effect = cv2.imread(path + 'effect_' + sensor + '_' + str(j) + '.png')
                for i in range(len(sensornames)):
                    sensor = sensornames[i]
                    path = '/mnt/my_dataset/second_affordance_dataset/' + objectname + '/' + toolname + '/' + action + '/' + sensor + '/'
                    init = resize(init, width, height)
                    effect = resize(effect, width, height)
                    init = bgr_to_rgb(init)
                    effect = bgr_to_rgb(effect)
                    init = normalize(init)
                    effect = normalize(effect)
                    init = to_float(init)
                    effect = to_float(effect)
                    
    