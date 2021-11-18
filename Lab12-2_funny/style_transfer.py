#!/usr/bin/env python
# coding: utf-8

# Hyper parameter definition
EPOCHS=10
STEP_PER_EPOCH=100
STYLE_WEIGHT=0.2
CONTENT_WEIGHT=1
TOTAL_VARIATION_WEIGHT=0.2
MAX_DIM = 512

# Paths
CONTENT_PATH = './input/mayuchi.jpg'
STYLE_PATH = './input/mnls.jpeg'
OUTPUT_DIR = './output/'
OUTPUT_NAME = 'style_transfer_mayuchi_'

# Content layer where will pull our feature maps
CONTENT_LAYERS = ['block5_conv2'] 

# Style layer of interest
STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

# In[1]:

import functools
import time
import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import IPython.display as display
from pathlib import Path
from PIL import Image
from matplotlib import pyplot
from guided_backprop import GuidedBackprop
from tqdm.notebook import tqdm

tf.config.experimental_run_functions_eagerly(True)
tf.config.run_functions_eagerly(True)
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False


# In[2]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# In[3]:


if not os.path.exists("input") : os.mkdir("input")
if not os.path.exists("output") : os.mkdir("output")


# # Part I (A Neural Algorithm of Artistic Style)

# 1. Implement total variational loss. tf.image.total_variation is not allowed (10%).
# 
# 2. Change the weights for the style, content, and total variational loss (10%).
# 
# 3. Use other layers in the model (10%). You need to calculate both content loss and style loss from different layers in the model
# 
# 4. Write a brief report. Explain how the results are affected when you change the weights, use different layers for calculating loss (10%). Insert markdown cells in the notebook to write the report.

# ## 一. 前置作業
# 我們先引進兩張圖片以及pretrain好的VGG。

# In[4]:


def load_img(path_to_img):
    max_dim = MAX_DIM
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    # in order to use CNN, add one additional dimension 
    # to the original image
    # img shape: [height, width, channel] -> [batch_size, height, width, channel]
    img = img[tf.newaxis, :]
    
    return img


# In[5]:


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


# 利用路徑呼叫圖片。

# In[6]:


content_path = CONTENT_PATH
style_path = STYLE_PATH

content_image = load_img(content_path)
style_image = load_img(style_path)

plt.figure(figsize=(14,10))
plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')


# 接著呼叫由imagenet pretrain好的VGG19

# In[7]:


vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.summary()


# ## 二. Extract style and content
# ### 1. 定義一個style/content的抽取器
# 首先我們要定義我們希望從VGG的哪一層抽出content以及style。Content features就是直接將Content image餵進去神經網路後，在我們想指定的某一層抽出來。Style feature則是先指定神經網路中的某幾層（以我們VGG的例子是各個block的第一層），計算其在這幾層中feature的layer（channel）-wise correlation (又稱為gram matrix)。

# In[8]:


# Content layer where will pull our feature maps
content_layers = CONTENT_LAYERS

# Style layer of interest
style_layers = STYLE_LAYERS

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# 接著就可以開始準備model了。我們的model就是把VGG中名字與指定為content/style layer的那幾層抽出來，return成新的model。這個model是用來extract style還是content取決於他的參數給的是content_layers還是style_layers。

# In[9]:


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


# 另外還要定義gram matrix。

# In[10]:


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


# 我們把以上的東西組合成抽取器。這個抽取器由兩個神經網路組成：抽取style的神經網路以及抽取content的神經網路。

# In[11]:


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__() #Python的繼承？
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name:value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content':content_dict, 'style':style_dict}

# ## 三. 定義各種loss與train function
# ### 1. 定義三種loss
# 這裡我們要使用三種loss，第一種是content loss，第二種是style loss，第三種是total variation loss
# $$V(y)=\sum_i\sum_j|y_{i+1,j}-y{i,j}|+|y_{i,j+1}-y_{i,j}|$$
# 實際訓練時我們使用的是三者加起來
# $$\mathscr{L}=\alpha\mathscr{L}_\text{content}+\beta\mathscr{L}_\text{style}+\gamma\mathscr{L}_\text{total variation}$$
# style_content_loss已經幫我們寫好了。

# In[14]:


def style_content_loss(outputs, style_weight, content_weight):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss



def total_variation_loss(image):
    [batch, height, width, channel] = image.get_shape().as_list()
    loss1 = tf.reduce_sum(tf.abs(image[:, 1:, :, :] - image[:, :height-1, :, :]))
    loss2 = tf.reduce_sum(tf.abs(image[:, :, 1:, :] - image[:, :, :width-1, :]))
    return tf.add(loss1, loss2)


# ### 2. train function
# 為了方便training，我直接寫一個train function可以接受不同的optimizer，各種loss之weight。另外，為了不讓梯度爆炸，先強制讓梯度的大小困在0與1之間。

# In[16]:


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# In[17]:


@tf.function()
def my_train_step(image, opt, style_weight, content_weight, total_variation_weight):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_weight, content_weight)
        loss += total_variation_weight*total_variation_loss(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


# In[18]:


def train_style_transfer(image, 
                         opt=tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1),
                         epochs=10, #要多少epoch? 一個epoch show 1張圖
                         steps_per_epoch=100, #每個epoch要train多少step
                         style_weight=0.1, 
                         content_weight=0.1, 
                         total_variation_weight=0):
    start = time.time()

    epochs = epochs
    steps_per_epoch = steps_per_epoch

    step = 0
    for n in tqdm(range(epochs)):
        for m in tqdm(range(steps_per_epoch)):
            step += 1
            my_train_step(image, opt, style_weight, content_weight, total_variation_weight)
        imshow(image.read_value())
        file_name = OUTPUT_DIR + OUTPUT_NAME + str(step) + '_epoch.png'
        mpl.image.imsave(file_name, image[0].numpy())
        #plt.title("Train step: {}".format(step))
        #plt.show()

    end = time.time()
    print("Total time: {:.1f}".format(end-start))
    return image


# ## 四. 改變不同weight的研究
# 首先引入圖片以及定義好我們要用的optimizer。

# In[19]:
# 爾後若要更改style_layers, content_layers時，先直接重新定義那兩個變數，以使用VGG的不同layer，再使用
# 
# extractor = StyleContentModel(style_layers, content_layers)
# 
# 呼叫即可。我們讓抽取器吃一張圖片看看能不能正常運作。

# In[12]:


extractor = StyleContentModel(style_layers, content_layers)
results = extractor(tf.constant(content_image))
style_results = results['style']

print('Styles:')
for name, output in sorted(results['style'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()

print("Contents:")
for name, output in sorted(results['content'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())


# In[13]:


style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

image = tf.Variable(content_image)
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


image1= train_style_transfer(image, 
                             opt=opt,
                             epochs=EPOCHS, 
                             steps_per_epoch=STEP_PER_EPOCH,
                             style_weight=STYLE_WEIGHT, 
                             content_weight=CONTENT_WEIGHT, 
                             total_variation_weight=TOTAL_VARIATION_WEIGHT)
