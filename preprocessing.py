import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import gc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm


sensornames = ['color', 'icub_left', 'icub_right']
toolnames = ['hook', 'ruler', 'spatula', 'sshot']
actions = ['left_to_right', 'pull', 'push', 'right_to_left']
objectnames = ['0_woodenCube', '1_pearToy', '2_yogurtYellowbottle', '3_cowToy', '4_tennisBallYellowGreen',
               '5_blackCoinBag', '6_lemonSodaCan', '7_peperoneGreenToy', '8_boxEgg', '9_pumpkinToy',
               '10_tomatoCan', '11_boxMilk', '12_containerNuts', '13_cornCob', '14_yellowFruitToy',
               '15_bottleNailPolisher', '16_boxRealSense', '17_clampOrange', '18_greenRectangleToy', '19_ketchupToy']

width, height = 128, 128

### FUNCTIONS ###
def stack_with_tools(image):
    # Stack the tools
    ruler = resize(cv2.imread("ruler_gray.png", cv2.IMREAD_GRAYSCALE), width, height)
    spatula = resize(cv2.imread("spatula_gray.png", cv2.IMREAD_GRAYSCALE),width, height)
    sshot = resize(cv2.imread("sshot_gray.png", cv2.IMREAD_GRAYSCALE),width, height)
    hook = resize(cv2.imread("hook_gray.png", cv2.IMREAD_GRAYSCALE),width, height)
    return np.stack((image,hook,ruler, spatula, sshot), axis=2)

def concatenate(image_1, image_2):
    # Concatenate the image
    return np.concatenate((image_1, image_2), axis=1)

def stack_channel(image_1, image_2):
    return np.concatenate((image_1, image_2), axis=2)

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

def stack_with_tools(image):
    # Stack the tools
    ruler = normalize(np.expand_dims(resize(cv2.imread("ruler.png", cv2.IMREAD_GRAYSCALE), width, height), axis=-1))
    spatula = normalize(np.expand_dims(resize(cv2.imread("spatula.png", cv2.IMREAD_GRAYSCALE),width, height), axis=-1))
    sshot = normalize(np.expand_dims(resize(cv2.imread("sshot.png", cv2.IMREAD_GRAYSCALE),width, height), axis=-1))
    hook = normalize(np.expand_dims(resize(cv2.imread("hook.png", cv2.IMREAD_GRAYSCALE),width, height), axis=-1))
    # print(image.shape, hook.shape, ruler.shape, spatula.shape, sshot.shape)
    stacked_images = np.concatenate((ruler, spatula, sshot, hook), axis=2)
    return np.concatenate((image,stacked_images), axis=2)


training_set, validation_set, test_set = [], [], []
# Initialize label encoders and one-hot encoders
tool_le = LabelEncoder()
action_le = LabelEncoder()
tool_ohe = OneHotEncoder()
action_ohe = OneHotEncoder()

# Fit the label encoders and one-hot encoders
tool_le.fit(toolnames)
action_le.fit(actions)
tool_ohe.fit(tool_le.transform(toolnames).reshape(-1, 1))
action_ohe.fit(action_le.transform(actions).reshape(-1, 1))

# Initialize lists to store the labels for each set
training_labels_1, validation_labels_1, test_labels_1 = [], [], []
training_labels_2, validation_labels_2, test_labels_2 = [], [], []
training_labels_3, validation_labels_3, test_labels_3 = [], [], []
training_tool, validation_tool, test_tool = [], [], []
training_action, validation_action, test_action = [], [], []
for a in tqdm(range(len(objectnames))):
    objectname = objectnames[a]
    for y in range(len(toolnames)):
        toolname = toolnames[y]
        for x in range(len(actions)):
            action = actions[x]
            label_tool = tool_le.transform([toolname])[0]
            label_action = action_le.transform([action])[0]
            label_tool_action = label_tool * len(actions) + label_action
            onehot_tool = tool_ohe.transform(label_tool.reshape(-1, 1)).toarray()[0]
            onehot_action = action_ohe.transform(label_action.reshape(-1, 1)).toarray()[0]
            action = actions[x]
            label = actions.index(action)
            # Split into sets: 60% Training, 20% Validation, 20% Testing
            ids = np.random.choice(np.arange(10), 10, replace=False)
            training_ids, validation_ids, testing_ids = ids[0:6], ids[6:8], ids[8:10]
            # Loop through the number of repeats
            for j in range(10):
                sensor_images = []
                for i in range(len(sensornames)):
                    sensor = sensornames[i]
                    path = 'action_recognition_dataset/' + objectname + '/' + toolname + '/' + action + '/' + sensor + '/'
                    
                    if sensor == 'icub_right' or sensor == 'icub_left':
                        init = cv2.imread(path + 'init_color_' + sensor + '_' + str(j) + '.png')
                        effect = cv2.imread(path + 'effect_color_' + sensor + '_' + str(j) + '.png')
                    else:
                        init = cv2.imread(path + 'init_' + sensor + '_' + str(j) + '.png')
                        effect = cv2.imread(path + 'effect_' + sensor + '_' + str(j) + '.png')
                    
                    init = resize(init, width, height)
                    effect = resize(effect, width, height)
                    init = bgr_to_rgb(init)
                    effect = bgr_to_rgb(effect)
                    init = normalize(init)
                    effect = normalize(effect)
                    init = to_float(init)
                    effect = to_float(effect)
                    image = stack_channel(init, effect)
                    sensor_images.append(image)
                    
                    # Delete variables that are no longer needed
                    del init, effect, image
                    gc.collect()

                stacked_image = np.concatenate(sensor_images, axis=2)  # Stack the sensor images
                stacked_image = stack_with_tools(stacked_image)
                # stacked_image = np.expand_dims(stacked_image, axis=0)  # Add an extra dimension
            # rest of your code

                if j in training_ids:
                    training_set.append(stacked_image)
                    training_labels_1.append(label_tool)
                    training_labels_2.append(label_action)
                    training_labels_3.append(label_tool_action)
                    training_tool.append(onehot_tool)
                    training_action.append(onehot_action)
                    
                elif j in validation_ids:
                    validation_set.append(stacked_image)
                    validation_labels_1.append(label_tool)
                    validation_labels_2.append(label_action)
                    validation_labels_3.append(label_tool_action)
                    validation_tool.append(onehot_tool)
                    validation_action.append(onehot_action)
                elif j in testing_ids:
                    test_set.append(stacked_image)
                    test_labels_1.append(label_tool)
                    test_labels_2.append(label_action)
                    test_labels_3.append(label_tool_action)
                    test_tool.append(onehot_tool)
                    test_action.append(onehot_action)
                del stacked_image
                gc.collect()


torch.save(training_set, 'training_set.pt')

torch.save(training_tool, 'training_tool.pt')

torch.save(training_action, 'training_action.pt')

torch.save(training_labels_1, 'training_labels_1.pt')

torch.save(training_labels_2, 'training_labels_2.pt')

torch.save(training_labels_3, 'training_labels_3.pt')


# Save validation data
torch.save(validation_set, 'validation_set.pt')

torch.save(validation_tool, 'validation_tool.pt')

torch.save(validation_action, 'validation_action.pt')

torch.save(validation_labels_1, 'validation_labels_1.pt')

torch.save(validation_labels_2, 'validation_labels_2.pt')

torch.save(validation_labels_3, 'validation_labels_3.pt')


# Save test data
torch.save(test_set, 'test_set.pt')

torch.save(test_tool, 'test_tool.pt')

torch.save(test_action, 'test_action.pt')

torch.save(test_labels_1, 'test_labels_1.pt')

torch.save(test_labels_2, 'test_labels_2.pt')

torch.save(test_labels_3, 'test_labels_3.pt')


# image = validation_set[0]
# # Display the first three channels in RGB
# plt.figure(figsize=(10, 12))

# plt.subplot(5, 5, 1)
# plt.imshow(image[:, :, :3])
# plt.title('color_init')

# plt.subplot(5, 5, 6)
# plt.imshow(image[:, :, 3:6])
# plt.title('color_effect')

# plt.subplot(5, 5, 2)
# plt.imshow(image[:, :, 6:9])
# plt.title('icub_left_init')

# plt.subplot(5, 5, 7)
# plt.imshow(image[:, :, 9:12])
# plt.title('icub_left_effect')

# plt.subplot(5, 5, 3)
# plt.imshow(image[:, :, 12:15])
# plt.title('icub_right_init')

# plt.subplot(5, 5, 8)
# plt.imshow(image[:, :, 15:18])
# plt.title('icub_right_effect')

# plt.subplot(5, 5, 4)
# plt.imshow(image[:, :, 18],cmap='gray')
# plt.title('hook')

# plt.subplot(5, 5, 9)
# plt.imshow(image[:, :, 19],cmap='gray')
# plt.title('ruler')

# plt.subplot(5, 5, 5)
# plt.imshow(image[:, :, 20],cmap='gray')
# plt.title('spatula')

# plt.subplot(5, 5, 10)
# plt.imshow(image[:, :, 21],cmap='gray')
# plt.title('sshot')


# plt.show()