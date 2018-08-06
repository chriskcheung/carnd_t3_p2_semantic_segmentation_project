[//]: #carnd_t3_p2_semantic_segmentation_project

[//]: # (Image References)
[image1]: ./output/neural_net.png
[image2]: ./output/road1.png
[image3]: ./output/image_vs_gt_image.png
[image4]: ./output/results.png
[image5]: ./output/40epoch8batchMlite1.png
[image6]: ./output/40epoch8batchMlite2.png
[image7]: ./output/40epoch8batchMlite3.png
[image8]: ./output/40epoch8batchMlite4.png
[image9]: ./output/40epoch8batchMlite5.png
[image10]: ./output/40epoch8batchMlite6.png
[video6]: ./output/IMG_4661.TRIM.MOV


# Carnd - term 3 - project 2 - semantic segmentation project

### Overview
The goal of this project is to build and train a neural network that can identify the road of the vehicle it is travling on. The neural network is composed of: an encoder and a decoder. The encoder is basically a pre-trained VGG16 model followed by a 1x1 convlution layer, while the decoder contains the transposed layers to upsample the pixels of the object and the skip connection layers that retains the background information surround the object in the image.

![alt text][image1]Semantic Segmentation Neural Network Pipeline

Model is trained over 289 road images from [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) which also provides the corresponding ground true images that labels the road pixels vs background pixels

![alt text][image3]


Images will be processed by the neural network in order to produce an image with a green mask on the road the vechicle is traveling on. 

![alt text][image2]


### Project Repository
All resource are located in Udacity's project repository
[CarND-Semantic-Segmentatio](https://github.com/udacity/CarND-Semantic-Segmentation)


### Project Submission
All modified code including results are committed to my personal github page
[carnd_t3_p2_semantic_segmentation_project](https://github.com/chriskcheung/carnd_t3_p2_semantic_segmentation_project)


### Key Files
##### main.py
consists of key functions to support building, training and testing the nerual network pipeline: load_vgg(), layers(), optimize(), train_nn(), and run().

load_vgg() - downloads the pre-trained VGG16 neural net model and extracts its tensorflow graphs as well as its image layer, keep_prob layer, pooling layer 3, 4, and 7. 

layers() - takes the extracted pooling layers from VGG16 model, connecting each following the Semantic Segmentation Nueral Network Pipeline diagram as shown above. Each layer in the decoder section is upsampled in order to bring the pixels of the object back to its original image width and length. We will go over the detail in the later section. 

optimize() - uses cross_entrophy_loss function to calculate the loss between trained image and its ground true image. Adam optimizer is used to minimumize loss during training.

train_nn() - trains the neural network by getting batches of images and labals by calling get_batches_fn(batch_size) function from helper.py, and feeding them into train_op and entropy_loss.

run() - put together a processing pipeline with the functions above. After training a model, a set of testing images are feed through the semantic segmentation neural network and the results are saved into runs directory for inspection.

##### helper.py
maybe_download_pretrained_vgg() - detects if VGG16 pre-trained model is not previously downloaded, it will download and install it accordingly

gen_batch_function() - a generate function that creates batches of training images base on batch_size provides

gen_test_output() - generates test output using test images

save_inference_samples() - as its name says, it feeds the test images obtain from gen_test_output() and saves the result of the semantic segmentation


## Project Setup
Due to the limitation of my PC laptop, I requested an AWS GPU instance to perform my semantic segmentation neural network training and testing. The project needs the a few more Tensorflow framework and Python packages to be installed than those mentioned in lecture. Road dataset comes from [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php).

    Python 3, TensorFlow-gpu, NumPy, SciPy, TQDM, Pillow

Details on how to run this project is documented in [CarND-Semantic-Segmentatio](https://github.com/udacity/CarND-Semantic-Segmentation).


## Implementation Challenge

### VGG16 Pre-trained Model
To take advantage of the existing Pretrained Model, we use VGG16 Neural Network, which can be downloaded [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).

Before using VGG16, it is better to understand each of its pooling layers size, in order for skip connections to match the layers they are connecting. In this project, we use its vgg_layer_3, vgg_layer_4, and vgg_layer_7, and each has an output class size of 256, 512, and 4096, respectively. The lecture never really talks about their width and length. This is actually dictated by the input_image tensor shape, which is based on the training image width and length. 

```python
    image_shape = (160, 576)
    input_image   = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], num_classes))
    correct_label = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], num_classes))
```

### Semantic Segmentation Neural Network Pipeline Implementation
One thing worth pointing out is, there are multiple ways of constructing the pipeline. I tried two and did some comparison before finalizing to one approach. I basically applied 1 x 1 convulation on vgg_layer7_out to a 1 x 1 x num_classes layer, then upsample the resulting layer using transposed convulation with stride of 2 as to upsample the result by 2 times. I added the upsample layer with vgg_layer4_out as skip connection to retain background information from the original vgg_layer4_out. The same step from upsampling and skip connection was repeated with vgg_layer3_out instead. 

The trick to match the output size of two layers during the additional of them for skip connection is to make sure the output of previous convolution layers matches the next one in the additional, by using the next vgg layer's shape[3] as the num_classes output during convolution. 

Stride size controls how much a layer can be upsampled. Stride of 2 upsample the width and heigth of the pixel by 2 times. Same goes on as stride of 8 upsamples the pixel width and height by 8 times. This model is based on a series of upsample sequence of 2 times, 2 times, and 8 times at each VGG layers that we used.

```python
    #taking the last layer of vgg16 to build a 1x1 fully convolutional layer, 
    #VGG layer tensors with 4D size [batch_size, original_height, original_width, num_classes]
    k_size = 1  # kernal_size
    stride = 1
    fconv8 = tf.layers.conv2d(vgg_layer7_out, 
                              num_classes, 
                              k_size, 
                              stride, 
                              padding='same', 
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    #upsampling 2 times from ?xHxWx4096 to ?x2Hx2Wx512
    #as upsampling is dictated by stride, adjust stride to 2
    k_size = 4  # kernal_size
    stride = 2
    upscl9 = tf.layers.conv2d_transpose(fconv8, 
                                        vgg_layer4_out.shape[3], 
                                        k_size, 
                                        stride, 
                                        padding='same', 
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    #skip connections by adding
    skip10 = tf.add(upscl9, vgg_layer4_out)

    #upsampling 2 times from ?x2Hx2Wx512 to ?x4Hx4Wx256
    k_size = 4  # kernal_size
    stride = 2
    upscl11 = tf.layers.conv2d_transpose(skip10, 
                                         vgg_layer3_out.shape[3], 
                                         k_size, 
                                         stride, 
                                         padding='same', 
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    #skip connections by adding
    skip12 = tf.add(upscl11, vgg_layer3_out)

    #upsampling 8 times from ?x4Hx4Wx256 to ?x32Hx32Wx2
    k_size = 16  # kernal_size
    stride = 8
    upscl13 = tf.layers.conv2d_transpose(skip12, 
                                         num_classes, 
                                         k_size, 
                                         stride, 
                                         padding='same', 
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    return upscl13
```

_Alternative_ pipeline implemenation turns all vgg layers output into output classes of 2 by applied 1 x 1 convolution. This keeps every layers of the pipeline to deal with the same output classes instead of needing to worry about using vgg_layer3_out.shape[3] and vgg_layer4_out.shape[3] at upsampling layers. This also works but the results show diffused edges of the green mask rather than solid edges. 


### Training
I started with a small epochs number, 5, and a small batch_size, 10, to get any idea of how the model behaves. The early trails found that the segmentation fell into the area of the road on the image. However, some output images showed the edges of the green mask was not fully extended to the edge of the road, while some output images showed the green mask sometimes covered areas that it shouldn't covered, like car, pedestrian walkway, etc.

Adjusting the batch_size, keep_prob, and epochs seems to be improve the trained model. Here are a few setting tried

![alt_text][image4]


### Results
Most of the info was provided in the class material and the walktrough video. The results still not the best at the current implementation, but it can be improved with larger training dataset, as well as picking cleaning dataset, and updating batch_size. Below, I capture a few screenshots to illustrate how my Semantic Segmentation Neural Network performs over 300 images.
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

