# Trustworthy Medical Segmentation with Uncertainty Estimation
# VMP Implementation for Lungs DATA 
# U-NET structure with 3 encoding and 2 decoding blocks
# Giuseppina Carannante

import tensorflow as tf
from tensorflow import keras
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import time, sys
import pickle
import timeit
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage
import pandas as pd
plt.ioff()
from mpl_toolkits import axes_grid1

def softplus(x):
    return np.log(1 + np.exp(x))   
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def save_adversarial_uncertainty(path,truex,adv, logits, truey,sigma,masked, images_n, Adversarial = True, targeted=True):
    path_full= path + './test_images/'
    if not os.path.exists(path_full):
        os.makedirs(path_full)
    N = 750 #number of test images to consider
    predict = np.argmax(logits, axis = -1)
    predict2 = predict
    uncert= np.take_along_axis(sigma, np.expand_dims(predict, axis=-1), axis=-1).squeeze(axis=-1)
    
    mean_u = np.mean(uncert)
    np.random.seed(70) 
    ind=np.random.choice(np.arange(N), images_n)
    
    cMap = []
    for value, colour in zip([0, 1],["Black", "Yellow"]): cMap.append((value/1., colour))
    customColourMap = LinearSegmentedColormap.from_list("custom", cMap)
    for i in ind:
        u = uncert[i,:,:]
        if Adversarial:
         M = masked[i,:,:]
        P = predict[i,:,:]
        L = truey[i,:,:]
        if Adversarial:
            plt.figure()
            plt.imshow(np.squeeze(truex[i,:,:]), 'gray', interpolation='none')
            plt.imshow(np.squeeze(adv[i,:,:]),'gray', interpolation='none', alpha= 0.8)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.savefig(path_full + '{}_Adversarial_noise.png'.format(i))
            plt.close()
        
        plt.figure(figsize=(10,10))
        plt.imshow(L, customColourMap, interpolation='none')
        plt.title("Ground truth Label") 
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(path_full + '{}_Label_image.png'.format(i))
        plt.close()


        plt.figure(figsize=(10,10))
        plt.imshow(P, customColourMap, interpolation='none')
        plt.title("Predicted Label") 
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(path_full + '{}_Predicted_image.png'.format(i))
        plt.close()

        plt.figure(figsize=(10,10))
        im= plt.imshow(u, cmap='winter_r', interpolation='nearest')
        plt.title("Uncertainty map") 
        add_colorbar(im)
        #plt.clim(m_un,mx_un)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(path_full +'{}_uncertainty_heatmap.png'.format(i))
        plt.close()

        if Adversarial:
         if targeted:
            plt.figure(figsize=(10,10))
            plt.imshow(M, customColourMap, interpolation='none')
            plt.title("Masked Label") 
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.savefig(path_full + '{}_Masked_Label_image.png'.format(i))
            plt.close()

    
    # creating uncertainty file to record the predictive variance:
    ##task 1: Everything vs Lung 
    mask1ant  = np.ma.masked_where(predict2 != 1, uncert) # masking all non-anterior
    anterior_unc = np.mean(mask1ant)
    mask2ant  = np.ma.masked_where(predict2 == 1, uncert) # masking  anterior 
    no_ant_unc = np.mean(mask2ant)
    
    textfile = open(path + 'Predictive_variance_tasks.txt','w')
    textfile.write('\n Average Predictive variance : ' +str(mean_u)) 
    textfile.write("\n---------------------------------")
    textfile.write('\n Predictive variance for lungs structures : ' +str(anterior_unc))         
    textfile.write('\n Predictive variance for background : ' +str(no_ant_unc)) 
    textfile.close()
    print('done saving uncertainty and creating images')
    return mean_u,  anterior_unc,  no_ant_unc
######################################################################
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%

def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
############################################
# function to compute log base 10 - used for SNR
def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator
############################################
# function to compute the DSC 
def dice(y_true, y_pred):
    true_mask, pred_mask = y_true, y_pred
    A=np.sum(true_mask, axis=(1,2))
    B=np.sum(pred_mask, axis=(1,2))
    im_sum = A+ B

    # Compute Dice coefficient
    intersection = np.multiply(true_mask, pred_mask)
    intersection_sum = np.sum(intersection, axis=(1,2))
    c=2. * intersection_sum / im_sum
    c_masked = np.ma.masked_invalid(c)
    c_masked_avr =np.mean(c_masked)
    var = np.var(c_masked)
    return c_masked_avr,var

def mask_tumor(y_true, y_pred):
    A,B = y_true, y_pred.numpy()
    di,all_di=dice(A,B)
    return di,all_di
######################################################################
#function colled by the upsampling layer to expand each slice of the 
#incoming tensor.
def unpool(value, name='unpool'):
    """
    :param value: A Tensor of shape [b, w,h, ch]
    :return: An upsampled Tensor of shape [b, 2*w+1, 2*h+1, ch]
    -> the output tensor will have alternating zeros and padded.
    e.g.
    input:      1 2 
                3 4

    output:  0 0 0 0 0
             0 1 0 2 0
             0 0 0 0 0
             0 3 0 4 0
             0 0 0 0 0
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out_before_pad = tf.reshape(out, out_size, name=scope)
        paddings = tf.constant([[0, 0,], [1, 0], [1, 0], [0, 0,]])
        out=tf.pad(out_before_pad, paddings, "CONSTANT")
    return out
####################################################
# Function to Pool a tensor using idices from another max pooling:   
def get_pooled(indices,other_tensor):
    #inputs: 
    #indeces: from pooling the mean tensor
    #other_tensor: (sigma) tensor to pool
    # Returns pooled sigma (other_tensor_pooled) 
    b = other_tensor.shape[0]
    w = other_tensor.shape[1]
    h=other_tensor.shape[2]
    c =other_tensor.shape[3]
    other_tensor_pooled = tf.gather(tf.reshape(other_tensor,shape= [b*w*h*c,]),indices)
    return other_tensor_pooled
##############################################################################
def expand_to_shape(data, shape, border='CONSTANT'):
    """
    Expands the array to the given image shape by padding it with a border (expects a tensor of shape [batches, nx, ny, channels].
    :param data: the array to expand
    :param shape: the target shape
    :param border: default CONSTANT -> alternatives: "SYMMETRIC" , "REFLECT"(probably not useful)
    """
    diff_nx = shape[1] - data.shape[1]
    diff_ny = shape[2] - data.shape[2]
    #NEEDED IF LEFT & RIGHT (TOP & BOTTOM) ARE DIFFERENT
    offset_nx_left = diff_nx // 2
    offset_ny_left = diff_ny // 2
    offset_nx_right = diff_nx - offset_nx_left
    offset_ny_right = diff_ny - offset_ny_left
    paddings = tf.constant([[0, 0,], [offset_nx_left, offset_nx_right], [offset_ny_left,offset_ny_right], [0, 0,]])
    expanded = tf.pad(data,paddings, border )
    return expanded
##################################################################################
# Function to crop tensor x1 according to shape given by tensor x2
def crop_tensor(x1,x2):
    with tf.name_scope("crop"):
        x1_shape = tf.shape(x1) # tensor coming from encoder path
        x2_shape = tf.shape(x2) # corresponding tensor from decoder
        # offsets for the top left corner of the crop:
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return x1_crop
#########################################################################
# This function computes the gradient of the ReLU function 
def grad_ReLU(mu_in):
  with tf.GradientTape() as g:
    g.watch(mu_in)
    out = tf.nn.relu(mu_in)
  gradi = g.gradient(out, mu_in) 
  return gradi
#################################################################################
# The class that propagates the mean and covariance matrix of the variational distribution 
# through the First Convolution Layer     
class myConv_input(tf.keras.layers.Layer):
    """y = Conv(x, w )"""
    """ x is constant and w is a r.v."""
    def __init__(self, kernel_num = 128,kernel_size=3, kernel_stride=1, padding="VALID",
                    mean_mu=0,mean_sigma=0.1, sigma_min=-12, sigma_max=-4.6):
        super(myConv_input, self).__init__()
        self.kernel_size = kernel_size #param kernel_size:  Size of the conv, kernel (Default   3)
        self.kernel_num = kernel_num #param out_channels: Number of Kernels (Required)
        self.kernel_stride = kernel_stride #param stride: Stride Length (Default   1)
        self.padding = padding 
        self.mean_mu =mean_mu
        self.mean_sigma=mean_sigma
        self.sigma_min =sigma_min
        self.sigma_max = sigma_max

    def build(self, input_shape):
        tau = 1.
        MYinitializer = tf.keras.initializers.TruncatedNormal(mean=self.mean_mu, stddev=self.mean_sigma) 
        dim = self.kernel_size * self.kernel_size
        self.w_mu = self.add_weight(name='w_mu1',
            shape=(self.kernel_size,self.kernel_size, input_shape[-1], self.kernel_num),
            initializer=MYinitializer, regularizer=tf.keras.regularizers.l2(tau),
            trainable=True
        )  
        self.w_sigma = self.add_weight(name = 'w_sigma1',
            shape=(self.kernel_num,),
            initializer=tf.random_uniform_initializer(minval= self.sigma_min, maxval=self.sigma_max,  seed=None), regularizer=sigma_regularizer(dim),
            trainable=True
        )   
                  
    def call(self, inputs):
        mu_out = tf.nn.conv2d(inputs, self.w_mu, strides=self.kernel_stride, padding=self.padding )
        w_sigma = tf.math.softplus(self.w_sigma)
        vect_sigma=tf.broadcast_to(w_sigma,[self.kernel_size*self.kernel_size*self.w_mu.shape[2], self.kernel_num], name=None)
        x_patches = tf.image.extract_patches(inputs, sizes=[1, self.kernel_size, self.kernel_size, 1], strides=[1,self.kernel_stride,self.kernel_stride,1], rates=[1,1,1,1], padding = self.padding)
        x_matrix = tf.reshape(x_patches,[inputs.shape[0], -1, self.kernel_size*self.kernel_size*self.w_mu.shape[2]]) #NON SICURA DELLA SHAPE w_mu[2]
        sigma=tf.matmul(tf.math.square(x_matrix),vect_sigma)
        sigma_out=tf.reshape(sigma, mu_out.shape)
        return mu_out, sigma_out
###################################################
# The class that propagates the mean and covariance matrix of the variational distribution 
# through the Intermediate Convolution Layers     
class myConv_intermediate(tf.keras.layers.Layer):
    """y = Conv(x, w )"""
    """ x and w are both r.v.
        :param kernel_num: Number of output channels       (Required)
        :param kernel_size:  Size of the conv, kernel        (Default   3)
        for 1x1 Convolution: change kernel_size = 1 and kernel_num = num_classes
        sigma_max= -2.2 (was 2.3 for MNIST)    
    """

    def __init__(self, kernel_num = 64 ,kernel_size=3, kernel_stride=1, padding="VALID",
                    mean_mu=0,mean_sigma=0.1, sigma_min=-12, sigma_max=-2.3):
        super(myConv_intermediate, self).__init__()
        self.kernel_size = kernel_size #param kernel_size:  Size of the conv, kernel (Default   3)
        self.kernel_num = kernel_num #param out_channels: Number of Kernels (Required)
        self.kernel_stride = kernel_stride #param stride: Stride Length (Default   1)
        self.padding = padding 
        self.mean_mu =mean_mu
        self.mean_sigma=mean_sigma
        self.sigma_min =sigma_min
        self.sigma_max = sigma_max

    def build(self, input_shape):
        #print(input_shape)
        tau = 1.
        MYinitializer =tf.keras.initializers.TruncatedNormal(mean=self.mean_mu, stddev=self.mean_sigma)
        dim = self.kernel_size * self.kernel_size
        self.w_mu = self.add_weight(name='w_mu',
            shape=(self.kernel_size,self.kernel_size, input_shape[-1], self.kernel_num),
            initializer=MYinitializer, regularizer=tf.keras.regularizers.l2(tau),
            trainable=True
        )  
        self.w_sigma = self.add_weight(name = 'w_sigma',
            shape=(self.kernel_num,),
            initializer=tf.random_uniform_initializer(minval= self.sigma_min, maxval=self.sigma_max, seed=None), regularizer=sigma_regularizer(dim), 
            trainable=True
        )   
                  
    def call(self, inputs, sigma_input):
        mu_out = tf.nn.conv2d(inputs, self.w_mu, strides=self.kernel_stride, padding=self.padding )
        w_sigma = tf.math.softplus(self.w_sigma)
        vect_sigma=tf.broadcast_to(w_sigma,[self.kernel_size*self.kernel_size*inputs.shape[-1], self.kernel_num], name=None)
        x_patches = tf.image.extract_patches(inputs, sizes=[1, self.kernel_size, self.kernel_size, 1], strides=[1,self.kernel_stride,self.kernel_stride,1], rates=[1,1,1,1], padding = self.padding)
        sigma_patches =tf.image.extract_patches(sigma_input, sizes=[1, self.kernel_size, self.kernel_size, 1], strides=[1,self.kernel_stride,self.kernel_stride,1], rates=[1,1,1,1], padding = self.padding)
        x_matrix = tf.reshape(x_patches,[inputs.shape[0], mu_out.shape[1]*mu_out.shape[1], self.kernel_size*self.kernel_size*inputs.shape[-1]]) 
        sigma_matrix = tf.reshape(sigma_patches,[inputs.shape[0], mu_out.shape[1]*mu_out.shape[1], self.kernel_size*self.kernel_size*inputs.shape[-1]])
        sigma1=tf.matmul(tf.math.square(x_matrix),vect_sigma)
        w_mean=tf.reshape(self.w_mu,[-1,inputs.shape[-1],self.kernel_num])
        w_mean=tf.reshape(w_mean,[-1,self.kernel_num])
        sigma2=tf.matmul(sigma_matrix, tf.math.square(w_mean))
        sigma3=tf.matmul(sigma_matrix,vect_sigma)
        sigma=sigma1+sigma2+sigma3 
        sigma_out=tf.reshape(sigma, mu_out.shape) 
        
        return mu_out, sigma_out
##########################################################
#propagates mean and covariance through the upsampling layer 
class myupsampling(keras.layers.Layer):
    """My_Upsampling"""

    def __init__(self):
        super(myupsampling, self).__init__()
    def call(self, mu_in, sigma_in):
        mu_out = unpool(mu_in)
        sigma_out = unpool(sigma_in)
        return mu_out, sigma_out
##############################################
#propagates mean and covariance through a padding layer (we pad mean and sigma in decoder)
class mypadding(keras.layers.Layer):
    """My_Padding"""

    def __init__(self, pad_size = [2,2], sigma_fill = 0, mode= "CONSTANT"):
        super(mypadding, self).__init__()
        self.size = pad_size
        self.sigma =sigma_fill
        self.mode = mode
    def call(self, mu_in, sigma_in):
        paddings = tf.constant([[0, 0,], self.size, self.size, [0, 0,]])
        mu_out = tf.pad(mu_in,paddings, self.mode )
        sigma_out = tf.pad(sigma_in,paddings, self.mode,  constant_values=self.sigma ) 
        return mu_out, sigma_out
##############################################
# Function to propagate mean and covariance through the pooling layer 
class mymaxpooling(keras.layers.Layer):
    """My_Max_Pooling"""

    def __init__(self):
        super(mymaxpooling, self).__init__()
    def call(self, mu_in, sigma_in):
        mu_out, argmax = tf.nn.max_pool_with_argmax(mu_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', include_batch_in_index=True) #shape=[1, new_size,new_size,num_filters[0]]  
        sigma_out= get_pooled(argmax,sigma_in)
        return mu_out, sigma_out
#############################################################
#Propagates mean and variance through the ReLU function 
class myReLU(keras.layers.Layer):
    """ReLU"""

    def __init__(self):
        super(myReLU, self).__init__()
    def call(self, mu_in, Sigma_in):
        mu_out = tf.nn.relu(mu_in)
        gradi = grad_ReLU(mu_in) 
        gradi_sq=tf.math.square(gradi)
        Sigma_out = tf.math.multiply(gradi_sq,Sigma_in)
        return mu_out, Sigma_out
############################################################################
# Function to Concatenate tensors from encoder and decoder paths  
class myConc(keras.layers.Layer):
    """Concatenation of downsampled (to be cropped) and upsampled feature maps
    inputs: mu_Decoder, Sigma_Decoder,mu_Encoder, Sigma_Encoder"""

    def __init__(self):
        super(myConc, self).__init__()
    def call(self, muD, SigmaD,muE, SigmaE):
        """
        Concatenation of Upsampled (from Decoder path) muD and SigmaD 
        with muE and SigmaE (corresponding Encoder tensors) respectively
        Args:
            inputs  muE and SigmaE (4-D Tensor): (N, H1, W1, C1), 
            inputs  muD and SigmaD (4-D Tensor): (N, H2, W2, C2)
        Returns:
            output (4-D Tensor): (N, H2, W2, C1 + C2)
        """
        mu_cropped = crop_tensor(muE,muD)
        sigma_cropped = crop_tensor(SigmaE,SigmaD)
        mu_out = tf.concat([muD, mu_cropped], axis=-1)
        sigma_out = tf.concat([SigmaD, sigma_cropped], axis=-1)
        return mu_out, sigma_out
#############################################################################
# The class that propagates the mean and covariance matrix of the variational distribution through the Softmax layer
class mysoftmax(keras.layers.Layer):
    """Mysoftmax pixel-wise"""

    def __init__(self):
        super(mysoftmax, self).__init__()
    def call(self, mu_in, sigma_in):
        mu_reshaped = tf.reshape(mu_in,[mu_in.shape[0], -1, mu_in.shape[3]])
        sigma_reshaped = tf.reshape(sigma_in,[sigma_in.shape[0], -1, sigma_in.shape[3]])
        mu_out = tf.nn.softmax(mu_reshaped)
        pp1 = tf.expand_dims(mu_out, axis=3)
        pp2 = tf.expand_dims(mu_out, axis=2)
        ppT = tf.matmul(pp1, pp2)
        p_diag = tf.linalg.diag(mu_out)
        grad = p_diag - ppT
        grad_sq = tf.math.square(grad)
        sigma_exp= tf.expand_dims(sigma_reshaped,axis=3)
        sigma = tf.matmul(grad_sq, sigma_exp)
        sigma_out= tf.squeeze(sigma)
        #mu_out and sigma_out shape = [bath, size*size, num_labels] 
        return mu_out, sigma_out

##########################################################################
# the log-likelihood of the objective function
# Inputs: 
#       y_pred_mean: The output Mean vector (predictive mean).
#       y_pred_sd: The output Variance vector (from predictive covariance matrix).
#       y_test: The ground truth prediction vector
# Output:
#       the expected log-likelihood term of the objective function  
def nll_gaussian(y_test, y_pred_mean, y_pred_sd): 

    eps=tf.constant(1e-3, shape=y_pred_sd.shape, name='epsilon')
    y_pred_sd_inv =  tf.math.reciprocal(tf.math.add(y_pred_sd, eps)) #Computes the reciprocal element-wise
    mu_square =  tf.math.square(y_pred_mean - y_test) 
    mu_square = tf.expand_dims(mu_square, axis=2)
    y_pred_sd_inv = tf.expand_dims(y_pred_sd_inv, axis=3) 
    loss1 =  tf.matmul(mu_square ,  y_pred_sd_inv) 
    loss = tf.reduce_mean(loss1, axis =0)
    loss = tf.reduce_mean(loss)
    #print('null gaussian loss1 ',loss)
    
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)
    
    loss2 = tf.math.reduce_prod(tf.math.add(y_pred_sd, eps), axis=-1) # shape [batch,size*size]
    loss2 = tf.math.log(loss2)
    loss2 = tf.reduce_mean(loss2) # not sure if I should do reduce mean
    #print('loglike part2', loss2)
    final_loss = 0.5*(loss+loss2)
    return final_loss
########################################IMPLEMENTED#######################
# This function computes the sigma regulirizer for the KL-divergence for convolution layers.
class sigma_regularizer(keras.regularizers.Regularizer):
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, x):
        f_s = tf.math.softplus(x) 
        return -self.strength * tf.reduce_mean(1. + tf.math.log(f_s) - f_s , axis=-1)
################################################################
# This class defines the netwrok by calling all layers          
class Density_prop_with_pad_UNET(tf.keras.Model):
  """ Required: number_of_kernels and number_of_labels"""

  def __init__(self, n_kernels, n_labels, name=None):
    super(Density_prop_with_pad_UNET, self).__init__()

    
    self.n_kernels = n_kernels
    self.n_out = n_labels
    self.conv_input = myConv_input(kernel_num = self.n_kernels)  
    self.conv1 = myConv_intermediate(kernel_num = self.n_kernels) 

    self.conv2 = myConv_intermediate(kernel_num = self.n_kernels*2)  
    self.conv3 = myConv_intermediate(kernel_num = self.n_kernels*2)

    self.conv4 = myConv_intermediate(kernel_num = self.n_kernels*4)  
    self.conv5 = myConv_intermediate(kernel_num = self.n_kernels*4)

    #  decoder path

    self.up1_conv2x2 = myConv_intermediate(kernel_num = self.n_kernels*2, kernel_size = 2, sigma_min=-4.6, sigma_max=-2.2)  
    self.up1_conv1 = myConv_intermediate(kernel_num = self.n_kernels*2)
    self.up1_conv2 = myConv_intermediate(kernel_num = self.n_kernels*2)
    
    self.up2_conv2x2 = myConv_intermediate(kernel_num = self.n_kernels, kernel_size = 2, sigma_min=-4.6, sigma_max=-2.2)  
    self.up2_conv1 = myConv_intermediate(kernel_num = self.n_kernels)
    self.up2_conv2 = myConv_intermediate(kernel_num = self.n_kernels)


    self.conv_final = myConv_intermediate(kernel_num = self.n_out, kernel_size = 1, sigma_min=-4.6, sigma_max=-2.2) #, sigma_min=-4.6, sigma_max=-2.2
    self.myrelu  = myReLU()
    self.myconc = myConc()
    self.mypad1 = mypadding(pad_size=[1,0], sigma_fill=0.05)
    self.mypad_up6 = mypadding(pad_size=[3,3],sigma_fill=0.05) 
    self.mypad = mypadding(sigma_fill=0.05)
    self.myups =  myupsampling()
    self.maxp =mymaxpooling()
    self.mysoftmax = mysoftmax()

  def call(self, inputs, training=True):
    # first block encoder:
    m1, s1 = self.conv_input(inputs) # 62
    m1, s1 = self.myrelu(m1, s1) # 
    m1, s1 =  self.conv1(m1, s1)  # 60
    m1, s1 = self.myrelu(m1, s1)  
    m1_p, s1_p = self.maxp(m1, s1) #  30
    # second block encoder:
    m2, s2 = self.conv2(m1_p, s1_p) # 28
    m2, s2 = self.myrelu(m2, s2) 
    m2, s2 =  self.conv3(m2, s2) # 26
    m2, s2 = self.myrelu(m2, s2)  
    m2_p, s2_p = self.maxp(m2, s2) # 13
    # third block encoder:
    m3, s3 = self.conv4(m2_p, s2_p) # 11
    m3, s3 = self.myrelu(m3, s3) 
    m3, s3 =  self.conv5(m3, s3) # 9
    m3, s3 = self.myrelu(m3, s3)  
    
    # first block decoder:
    d_m1, d_s1 = self.myups(m3, s3) 
    d_m1, d_s1 = self.up1_conv2x2(d_m1, d_s1) #18
    d_m1, d_s1 = self.mypad_up6(d_m1, d_s1) # 24
    d_m1, d_s1 = self.myconc(d_m1, d_s1, m2, s2)
    d_m1, d_s1 = self.up1_conv1(d_m1, d_s1 ) #22
    d_m1, d_s1 = self.myrelu(d_m1, d_s1)
    d_m1, d_s1 = self.mypad_up6(d_m1, d_s1) #26
    d_m1, d_s1 =  self.up1_conv2(d_m1, d_s1) #24
    d_m1, d_s1 = self.myrelu(d_m1, d_s1)
    # second block decoder:
    d_m2, d_s2 = self.myups(d_m1, d_s1) 
    d_m2, d_s2 = self.up2_conv2x2(d_m2, d_s2) #48
    d_m2, d_s2 = self.mypad_up6(d_m2, d_s2) #54
    d_m2, d_s2 = self.myconc(d_m2, d_s2, m1, s1)
    d_m2, d_s2 = self.up2_conv1(d_m2, d_s2) #52
    d_m2, d_s2 = self.myrelu(d_m2, d_s2)
    d_m2, d_s2 = self.mypad_up6(d_m2, d_s2)
    d_m2, d_s2 =  self.up2_conv2(d_m2, d_s2) #54
    d_m2, d_s2 = self.myrelu(d_m2, d_s2)
    
   
    m_final, s_final =  self.conv_final(d_m2, d_s2 )
    outputs, Sigma= self.mysoftmax(m_final, s_final) # output images are flattened (& sigma vector)
    #shape [batch, out_image_size*out_image_size, n_labels]
    return outputs, Sigma
def crop_to_wanted_shape(x, shape):
    """ function to crop input image to wanted shape:
    _______________________________________________________
    inputs: x = tensor to crop, shape = choosen dimension"""
    
    x1_shape = tf.shape(x) # tensor
    #x2_shape = tf.constant([x1_shape[0], shape,shape,x1_shape[-1]  ]) 
    # offsets for the top left corner of the crop:
    offsets = [0, (x1_shape[1] - shape) // 2, (x1_shape[2] - shape) // 2, 0]
    size = [-1,shape, shape, -1]
    x1_crop = tf.slice(x, offsets, size)
    return x1_crop
def crop_numpy_image(x,shape, num_ch = 3):
    """ function to crop input image to wanted shape when images are fed as numpy arrays:
    _______________________________________________________
    inputs: x = tensor to crop, shape = choosen dimension"""
    or_size = x.shape[1]
    start =(or_size - shape)/2 # starting point to slice image
    start = int(start)
    end_p = or_size - start
    end_p = int(end_p)
    
    if num_ch==3:
        im = x[:,start:end_p, start:end_p,:]
    else:
        im = x[:,start:end_p, start:end_p]
    return im
    
# Function to train network and build adversarial attacks
def main_function(n_kernels=16, output_channels = 2, batch_size=10, epochs = 5, lr=0.001, kl_factor = 0.005,
         Training = True, continue_training =True, saved_model_epochs=50, Adversarial_noise=False,  
        epsilon = 0.0001, Targeted=True, maxAdvStep=20, stepSize =1, adversary_targeted_class = 2 
        , adv_class = 3):  
    
    
    PATH = '/data/giuse/Segmentation_data/Covid2/saved_models_VMP/epoch_{}/'.format(epochs) 
    if not os.path.exists( PATH):
        os.makedirs(PATH)  	    
    output_size = output_channels 
    
    
    t2 = open('/data/giuse/Segmentation_data/Covid/train_test_binary.pkl', 'rb')
    x_train,y_train,x_test,y_test = pickle.load(t2)                                                               
    t2.close()
   
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    tr_dataset = (
    train_loader.shuffle(len(x_train))
    .batch(batch_size)
    .prefetch(2)
    )
    if Training:
    
        val_dataset = (
        validation_loader.shuffle(len(x_test))
        .batch(batch_size)
        .prefetch(2)
        )
    else:
        val_dataset = (
        validation_loader
        .batch(batch_size)
        .prefetch(2)
        )
    
    print('initializing model')    
    UNET_model = Density_prop_with_pad_UNET(n_kernels, output_size, name = 'vmp_unet') 
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    @tf.function  
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            logits, sigma = UNET_model(x)           
            loss_final = nll_gaussian(y, logits,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-12),
                                   clip_value_max=tf.constant(1e+3)))
            regularization_loss=tf.math.add_n(UNET_model.losses)            
            loss = loss_final + kl_factor* 0.5 *regularization_loss 
               
            gradients = tape.gradient(loss, UNET_model.trainable_weights)  
        optimizer.apply_gradients(zip(gradients, UNET_model.trainable_weights))       
        return loss, logits, sigma,  regularization_loss, loss_final

    @tf.function    
    def create_adversarial_pattern(input_image, input_label):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
            UNET_model.trainable = False 
            prediction, sigma = UNET_model(input_image) 
            loss_final = nll_gaussian(input_label, prediction,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+4),
                                   clip_value_max=tf.constant(1e+3)))                         
            loss = 0.5 * loss_final 
            
          # Get the gradients of the loss w.r.t to the input image.
          gradient = tape.gradient(loss, input_image)
          # Get the sign of the gradients to create the perturbation
          signed_grad = tf.sign(gradient)
          return signed_grad, loss
    if Training:
        if continue_training: 
            saved_model_path = '/data/giuse/Segmentation_data/Covid2/saved_models_VMP/epoch_{}/'.format(saved_model_epochs)
            UNET_model.load_weights(saved_model_path + 'vmp_UNET_model')
        train_acc = np.zeros(epochs) 
        valid_acc = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_error = np.zeros(epochs)
        start = timeit.default_timer()
        #############################
        train_dice1 =np.zeros(epochs)
        val_dice1 =np.zeros(epochs)
    
      

        for epoch in range(epochs):
            print('Epoch: ', epoch+1, '/' , epochs)           
            acc1 = 0
            acc_valid1 = 0 
            err1 = 0
            err_REG =0
            err_KL=0
            err_valid1 = 0
            ################
            d1=0
            val_d1 =0
            
            tr_no_steps = 0
            va_no_steps = 0           
            #-------------Training-------------------- 
            for step, (x, y) in enumerate(tr_dataset):                                          
              
                update_progress(step/int(x_train.shape[0]/(batch_size)) ) 
                x=tf.dtypes.cast(x, tf.float32)
                x = tf.expand_dims(x, -1)
                y = crop_numpy_image(y,508,1)
                y = tf.dtypes.cast(y, tf.int32)
                one_hot_y_train = tf.one_hot(y, depth=output_size)
                y_flatten = tf.reshape(one_hot_y_train,[one_hot_y_train.shape[0], -1, one_hot_y_train.shape[3]])            
                loss, logits, sigma, reg, kl = train_on_batch(x, y_flatten)                 
                err1+= loss
                err_REG += reg
                err_KL+=kl
                y_predictions = tf.math.argmax(logits, axis=2) 
                pred_reshape = tf.reshape(y_predictions, [y.shape[0], y.shape[1], y.shape[2]])
                y_arg = tf.math.argmax(y_flatten,axis=2)
                corr = tf.equal(y_predictions,y_arg)
                accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))                            
                acc1+=accuracy 
                di,di_all = mask_tumor(y,pred_reshape)
                d1+= di
               
                if step % 20 == 0:
                    print("\n Step:", step, "Loss:" , float(err1/(tr_no_steps + 1.)))
                    print("Total Training accuracy so far: %.3f" % float(acc1/(tr_no_steps + 1.)), 
                     " - New Dice Coefficients:" , float(d1/(tr_no_steps + 1.)))
                    
                           
                tr_no_steps+=1
            image_size = x.shape[1]         
            train_acc[epoch] = acc1/tr_no_steps
            train_err[epoch] = err1/tr_no_steps
            train_dice1[epoch] = d1/tr_no_steps
            
            
            print('Training new dice scores  ', train_dice1[epoch])
            UNET_model.save_weights(PATH + 'vmp_UNET_model')                  
    
            #---------------Validation----------------------           
            for step, (x, y) in enumerate(val_dataset):               
                update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
                
                x = tf.dtypes.cast(x, tf.float32)
                y_crop = crop_numpy_image(y,508,1)
                y_crop = tf.dtypes.cast(y_crop, tf.int32)
                y =  tf.one_hot(y_crop, depth=output_size) 
                x = tf.expand_dims(x,axis=-1)
                     
                logits, sigma = UNET_model(x)  
                
                y_flatten = tf.reshape(y,[y.shape[0], -1, y.shape[3]])            
                vloss = nll_gaussian(y_flatten, logits,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-12),
                                           clip_value_max=tf.constant(1e+3)))
                                           
                regularization_loss=tf.math.add_n(UNET_model.losses)
                total_vloss = vloss + kl_factor*0.5 *regularization_loss
                err_valid1+= total_vloss  
                y_predictions = tf.math.argmax(logits, axis=2) 
                y_arg = tf.math.argmax(y_flatten,axis=2)
                pred_reshape = tf.reshape(y_predictions, [y.shape[0], y.shape[1], y.shape[2]])
                corr = tf.equal(y_predictions,y_arg)                  
                va_accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                acc_valid1+=va_accuracy
                
                di,di_all = mask_tumor(y_crop,pred_reshape)
                val_d1+= di
                #################################################################################
                if step % 8 == 0:                   
                    print("Step:", step, "Loss:", float(total_vloss))
                    print("Total validation accuracy so far: %.3f" % float(acc1/(va_no_steps+ 1.)), 
                     " -  Dice Coefficient:" , float(val_d1/(va_no_steps+ 1.)))
                    
                va_no_steps+=1
          
            valid_acc[epoch] = acc_valid1/va_no_steps
            valid_error[epoch] = err_valid1/va_no_steps
            val_dice1[epoch] = val_d1/va_no_steps

            stop = timeit.default_timer()
            print('Total Training Time: ', stop - start)
            print('Training Acc  ', train_acc[epoch])
            print('Validation Acc  ', valid_acc[epoch])
            print('------------------------------------')
            print('Training error  ', train_err[epoch])
            print('Validation error  ', valid_error[epoch]) 
            
            print('------------------------------------')
            print('train dice score ', train_dice1[epoch])
            print('Validation dice score ', val_dice1[epoch]) 
            
        #-----------------End Training--------------------------             
        UNET_model.save_weights(PATH + 'vmp_UNET_model')        
        if (epochs > 1):
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_acc, 'b', label='Training acc')
            plt.plot(valid_acc,'r' , label='Validation acc')
            plt.ylim(0, 1.1)
            plt.title("Trustworthy Medical Segmentation with Uncertainty Estimation")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'VMP_UNET_Data_acc.png')
            plt.close(fig)    
    
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_err, 'b', label='Training error')
            plt.plot(valid_error,'r' , label='Validation error')
            #plt.ylim(0, 1.1)
            plt.title("Trustworthy Medical Segmentation with Uncertainty Estimation")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'VMP_UNET_Data_error.png')
            plt.close(fig)

            fig = plt.figure(figsize=(15,7))
            plt.plot(train_dice1, 'b', label='Training Dice T')
            plt.plot(val_dice1,'r' , label='Validation Dice T')
          
            
            #plt.ylim(0, 1.1)
            plt.title("Trustworthy Medical Segmentation with Uncertainty Estimation")
            plt.xlabel("Epochs")
            plt.ylabel("dice coefficient")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'VMP_UNET_Data_DICE.png')
            plt.close(fig)

            
        
        f = open(PATH + 'training_validation_acc_error.pkl', 'wb')         
        pickle.dump([train_acc, valid_acc, train_err, valid_error], f)                                                   
        f.close()                  
             
        textfile = open(PATH + 'Related_hyperparameters.txt','w')    
        textfile.write(' Image Dimension : ' +str(image_size))
        textfile.write('\n No kernels in first conv block : : ' +str(n_kernels))
        textfile.write('\n No Labels : ' +str(output_size))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Learning rate : ' +str(lr))    
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))         
        textfile.write("\n---------------------------------")          
        if Training: 
            textfile.write('\n Total run time in sec : ' +str(stop - start))
            if(epochs == 1):
                textfile.write("\n Averaged Training  Accuracy : "+ str( train_acc))
                textfile.write("\n Averaged Validation Accuracy : "+ str(valid_acc ))
                    
                textfile.write("\n Averaged Training  error : "+ str( train_err))
                textfile.write("\n Averaged Validation error : "+ str(valid_error ))

                                                   
            else:
                textfile.write("\n Averaged Training  Accuracy : "+ str(np.mean(train_acc[epoch])))
                textfile.write("\n Averaged Validation Accuracy : "+ str(np.mean(valid_acc[epoch])))
                
                textfile.write("\n Averaged Training  error : "+ str(np.mean(train_err[epoch])))
                textfile.write("\n Averaged Validation error : "+ str(np.mean(valid_error[epoch])))
              
                textfile.write("\n Averaged Training  dice score anterior: "+ str(np.mean(train_dice1[epoch])))
                textfile.write("\n Averaged Validation dice score anterior: "+ str(np.mean(val_dice1[epoch]))) 
                                                 
                                                   
        textfile.write("\n---------------------------------")                
        textfile.write("\n---------------------------------")    
        textfile.close()
    #-------------------------Testing-----------------------------    
    else:
        
        no_test_images=x_test.shape[0]
        t_d1=0
        image_size = 512
        out_image_size = 508
        num_channel = 1
        batch_SNR = 0
        if Targeted:
            test_path = 'target_{}_with_{}_adversarial_noise_{}_max_iter_{}_{}/'.format(adversary_targeted_class, adv_class, epsilon, maxAdvStep, stepSize)            
        else:
            test_path = 'test_results_non_targeted_adversarial_noise_{}/'.format(epsilon)              
        UNET_model.load_weights(PATH + 'vmp_UNET_model')       
        test_no_steps = 0        
        n_test_steps =int(x_test.shape[0] / (batch_size))
       
        all_dice_coef1 = []
        
        

        true_x = np.zeros([no_test_images, out_image_size, out_image_size, num_channel])
        adv_perturbations= np.zeros([no_test_images, out_image_size, out_image_size, num_channel])
        true_y = np.zeros([no_test_images, out_image_size, out_image_size])
        masked_y = np.zeros([no_test_images, out_image_size, out_image_size])
        logits_ = np.zeros([no_test_images,out_image_size, out_image_size, output_size])
    
        predicted_y = np.zeros([no_test_images, out_image_size, out_image_size])
        sigma_ = np.zeros([no_test_images,out_image_size, out_image_size, output_size])

        acc_test = 0
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)) )
            x = tf.dtypes.cast(x, tf.float32)
            x = tf.expand_dims(x,axis=-1) 
            
            x_crop = crop_numpy_image(x, out_image_size, 3)
            true_x[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = x_crop
            y_crop  = crop_numpy_image(y, out_image_size, 1)
            y_crop = tf.dtypes.cast(y_crop, tf.int32)
            true_y[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = y_crop
            y_arg = y_crop
            x_min = np.amin(x)
            x_max =np.amax(x)


            adv_x = x
              
            if Targeted:
             for advStep in range(maxAdvStep):  
                mask = np.ma.masked_where(y_crop==adversary_targeted_class , y_crop) 
                masked_label=np.ma.filled(mask, fill_value=adv_class)
                masked_y[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = masked_label
                y_true_batch =tf.one_hot(masked_label, depth=output_size)
                
                y_true_batch = tf.reshape(y_true_batch,[y_true_batch.shape[0], -1, y_true_batch.shape[3]])   
                adv_batch, adv_loss = create_adversarial_pattern( adv_x , y_true_batch)
                adv_batch2 = crop_numpy_image(adv_batch, out_image_size, 3)
                adv_perturbations[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :,:, :] = adv_batch2
             
                adv_x = adv_x + stepSize *adv_batch
                adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)
                adv_x = tf.clip_by_value(adv_x, x_min, x_max) 
                adv_x2 = crop_numpy_image(adv_x, out_image_size, 3) 
            
            
            else:
                y =  tf.one_hot(y_crop, depth=output_size) 
                y_flatten = tf.reshape(y,[y.shape[0], -1, y.shape[3]])
                adv_batch, adv_loss = create_adversarial_pattern(adv_x, y_flatten) 
                adv_batch2 = crop_numpy_image(adv_batch, out_image_size, 3)              
                adv_perturbations[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :,:, :] = adv_batch2 
                adv_x = adv_x + adv_batch
                adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)
                adv_x = tf.clip_by_value(adv_x, x_min, x_max) 
                adv_x2 = crop_numpy_image(adv_x, out_image_size, 3)
            
            #####################################################################
            
            start = timeit.default_timer()
            logits, sigma   = UNET_model(adv_x)
            stop = timeit.default_timer()  
            logits_im_shape = tf.reshape(logits,[logits.shape[0], out_image_size,out_image_size, logits.shape[2]])
            logits_[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ),:,:,:] =logits_im_shape
            pred =tf.math.argmax(logits, axis=2)
            sigma_im_shape = tf.reshape(sigma,[logits.shape[0], out_image_size,out_image_size, logits.shape[2]])
            sigma_[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :]= sigma_im_shape 
            y_crop=tf.one_hot(y_crop, depth=output_size)
            y_flatten = tf.reshape(y_crop,[y_crop.shape[0], -1, y_crop.shape[3]])
            y_arg2 =  tf.math.argmax(y_flatten,axis=2)  
            corr = tf.equal(tf.math.argmax(logits, axis=2),y_arg2)
            accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
            acc_test+=accuracy
            pred = tf.reshape(pred,[pred.shape[0], out_image_size,out_image_size])
            predicted_y[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = pred
            di,di_all = mask_tumor(y_arg,pred)
            all_dice_coef1.append(di_all)
            t_d1+= di
            
               
            if step % 10 == 0:
                print("Total running accuracy so far: %.3f" % accuracy.numpy())             
            test_no_steps+=1 

            
            numerator = np.sum(np.square(x_crop))/batch_size*num_channel
            diff = adv_x2 - x_crop
            denominator = np.sum(np.square(diff))/batch_size*num_channel
            sig = numerator/denominator
            snr_b = np.mean(10 * np.log10(sig))
            batch_SNR+= snr_b

            if step % 100 ==0:
                print('value of snr_b', snr_b)
            
        test_acc = np.mean(acc_test.numpy())  
        test_d1 = t_d1/test_no_steps
        print('Test accuracy : ', test_acc)  
        print('Test Dice Coefficient : ', test_d1) 
        
        test_std_d1 = np.sqrt(np.mean(all_dice_coef1))
     

        saved_result_path =  PATH + test_path 
        if not os.path.exists(saved_result_path):
            os.makedirs(saved_result_path)
        pf = open(saved_result_path + 'uncertainty_info.pkl', 'wb')            
        pickle.dump([predicted_y,logits_, sigma_,  adv_perturbations,masked_y, test_acc], pf)                                                
        pf.close()
        
        
        predicted_y = tf.cast(predicted_y, tf.int32)
        variance = np.take_along_axis(sigma_, np.expand_dims(predicted_y, axis=-1), axis=-1)
        variance = tf.math.reduce_mean(variance) 
        var = variance.numpy()
        print('Output Variance',var )
        print('SNR',batch_SNR/n_test_steps)         
        if Targeted:
            unc, class1_unc, class2_unc = save_adversarial_uncertainty(saved_result_path,true_x,epsilon*adv_perturbations,logits_, true_y,sigma_,masked_y,  10, Adversarial = True )
        else:
            unc, class1_unc, class2_unc = save_adversarial_uncertainty(saved_result_path,true_x,epsilon*adv_perturbations,logits_, true_y,sigma_,masked_y,  10, Adversarial = True, targeted=False )
        textfile = open(saved_result_path + 'Related_hyperparameters_adversarial.txt','w')   
  
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write("\n---------------------------------")
        textfile.write("\n Averaged Test Accuracy : "+ str( test_acc))  
        textfile.write("\n Output Variance: "+ str(np.mean(var)))                     
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.write("\n Averaged Test Dice Coefficient : "+ str( test_d1))
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.write("\n std Test Dice Coefficient : "+ str( test_std_d1))
       
        if Adversarial_noise:
            if Targeted:
                
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adv_class))  
                textfile.write('\n attacked class: ' + str(adversary_targeted_class))                                    
                textfile.write("\n---------------------------------")
                textfile.write('\n predictive Variance per class:') 
                
                textfile.write(str(class1_unc)) 
                textfile.write(str(class2_unc)) 
                   
                
            else:      
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write("\n Test Time in sec (per sample) : "+ str(stop - start))  
            textfile.write('\n Adversarial Noise epsilon: '+ str(epsilon ))    
            textfile.write("\n SNR: "+ str(batch_SNR/n_test_steps))               
        textfile.write("\n---------------------------------")    
        textfile.close()
  

if __name__ == '__main__':
    main_function()
    main_function(Training= False, Adversarial_noise = True, epsilon = 0.001, Targeted=False)
###################################################################################################
# Functions to test on noise Free Data and with added Gaussian noise

print('testing ')
def testing(epochs, labels, gaussain_noise_std, slices = 1, n_kernels = 16,Random_noise =False, noise_on = 'B'):
    """ function to test model with or without noise added."""
    print('starting testing')
    output_size = 2
    batch_size =10 # user defined 
    t2 = open('/data/giuse/Segmentation_data/Covid/train_test_binary.pkl', 'rb')
    x_train,y_train,x_test,y_test = pickle.load(t2)                                                               
    t2.close()
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    validation_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    val_dataset = (
    validation_loader
    .batch(batch_size)
    .prefetch(2)
    )
    UNET_model = Density_prop_with_pad_UNET(n_kernels, output_size, name = 'vmp_unet')
    
    PATH = '/data/giuse/Segmentation_data/Covid2/saved_models_VMP/epoch_{}/'.format(epochs) 
    
     
    if Random_noise:
            test_path = 'test_results_random_noise_{}/'.format(gaussain_noise_std)
    else:
            test_path = 'test_results_no_noise/'
           
    UNET_model.load_weights(PATH + 'vmp_UNET_model')
    
    no_test_images = x_test.shape[0]
    
    output_size = 2 # there are 2 labels to predict
    num_channel = slices
    
    acc_test = 0
    test_no_steps = 0
    t_d1=0
    
    new_snr = 0
    or_image_size = 512 # need this if output size is different from image size
    image_size = 508
    signal = 0
    true_x = np.zeros([no_test_images, image_size, image_size, num_channel])
    noisy_x = np.zeros([no_test_images, image_size, image_size, num_channel])
    true_y = np.zeros([no_test_images, image_size, image_size])
    logits_ = np.zeros([no_test_images,image_size, image_size, output_size])
    
    predicted_y = np.zeros([no_test_images, image_size, image_size])
    sigma_ = np.zeros([no_test_images,image_size, image_size, output_size])
    gauss_std = 0
    all_dice_coef1 = []
    
    for step, (x, y) in enumerate(val_dataset): 
               
            x = tf.dtypes.cast(x, tf.float32)
            x = tf.expand_dims(x,axis=-1) 
            # use next 2 lines if want image_size to be different
            x1 = crop_numpy_image(x,image_size, num_ch = 3)
            y1 = crop_numpy_image(y,image_size, num_ch = 0)
            ###############################
            update_progress(step / int(no_test_images/batch_size) )
            true_x[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = x1 # saving cropped images
            true_y[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = y1 # saving cropped labels
            
            y_arg = y
            y1 = tf.dtypes.cast(y1, tf.int32)
            y =  tf.one_hot(y1, depth=output_size) 
            t_x = x1 # noise free signal 
            snr = 0
            max_val = np.amax(x1)
            min_val = np.amin(x1)

            if Random_noise:
                noise = tf.random.normal(shape = [batch_size, or_image_size, or_image_size, num_channel], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
                if noise_on == 'A':
                    y_expanded=tf.expand_dims(y_arg, axis=3)
                    y_expanded = tf.broadcast_to(y_expanded,shape = [batch_size, or_image_size, or_image_size, num_channel] )
                    mask_1 = np.ma.masked_where(y_expanded == 0, noise) # true for background
                    mask_2=np.ma.filled(mask_1, fill_value=0)  # set noise = 0 wher corresponding to background
                    x = x +  mask_2
                    x= np.clip(x,min_val,max_val)
                
                else:
                    x = x +  noise # apply noise everywhere
                    x= np.clip(x,min_val,max_val)
                
                x_crop = crop_numpy_image(x, image_size)
                numerator = np.sum(np.square(t_x))/batch_size*num_channel
                diff = x_crop - t_x
                denominator = np.sum(np.square(diff))/batch_size*num_channel
                sig = numerator/denominator
                snr = np.mean(10 * np.log10(sig))
                
                noisy_x[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :, :] = x_crop
                
            logits, sigma = UNET_model(x) 
        
            sigma_im_shape = tf.reshape(sigma,[logits.shape[0], image_size,image_size, logits.shape[2]])
            logits_im_shape = tf.reshape(logits,[logits.shape[0], image_size,image_size, logits.shape[2]])
            
            logits = tf.reshape(logits_im_shape, [y.shape[0], -1, y.shape[3]])
            #################################################

            sigma_[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :]= sigma_im_shape 
            logits_[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ),:,:,:] =logits_im_shape
            pred =tf.math.argmax(logits, axis=2)

            y_flatten = tf.reshape(y,[y.shape[0], -1, y.shape[3]])
            y_arg2 =  tf.math.argmax(y_flatten,axis=2)            
            corr = tf.equal(pred,y_arg2)
            accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
           
            pred = tf.reshape(pred,[pred.shape[0], image_size,image_size])
            predicted_y[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = pred
            di,di_all = mask_tumor(y1,pred)
            all_dice_coef1.append(di_all)
            t_d1+= di
           
            acc_test+=accuracy
            new_snr += snr
            if Random_noise:
                signal+=sig
                                            
            test_no_steps+=1   
    gauss_std = gauss_std/test_no_steps

    test_acc = acc_test/test_no_steps
    new_t_snr = new_snr/test_no_steps 

    average_signal = signal/test_no_steps
    avr_sig = average_signal
    test_d1 = t_d1/test_no_steps
    
    test_std_d1 = np.sqrt(np.mean(all_dice_coef1))
    
    predicted_y = tf.cast(predicted_y, tf.int32)  ## make numpy array
    variance = np.take_along_axis(sigma_, np.expand_dims(predicted_y.numpy(), axis=-1), axis=-1).squeeze(axis=-1)
    variance = np.mean(variance)
    print('Test Dice Coefficients : ', test_d1) 
    data_folder =  '/data/giuse/Segmentation_data/Covid/saved_models_VMP/epoch_{}/'.format(epochs) 
    if Random_noise:
            print('Test average snr signal : ',new_t_snr)
            if noise_on == 'A':
                path_full= data_folder + test_path + './on_lungs/'
                if not os.path.exists(path_full):
                    os.makedirs(path_full)
              
                pf = open(data_folder + test_path + './on_lungs/uncertainty_info_on_lungs_noise_{}.pkl'.format(gaussain_noise_std), 'wb')
            else:
                path_full= data_folder + test_path + './on_all/'
                if not os.path.exists(path_full):
                    os.makedirs(path_full)
                pf = open(data_folder + test_path + './on_all/uncertainty_info_noise_{}.pkl'.format(gaussain_noise_std), 'wb')       
              
            pickle.dump([logits_, sigma_, noisy_x, true_y, test_acc.numpy() ], pf)                                                                           
            pf.close()
    else:
            path_full= data_folder + test_path
            if not os.path.exists(path_full):
                os.makedirs(path_full)
            pf = open(data_folder + test_path + 'uncertainty_info.pkl', 'wb')     
           
            pickle.dump([logits_, sigma_, true_x, true_y, test_acc.numpy() ], pf)   
            pf.close()
        
        
    if Random_noise: 
            if noise_on =="A":
                textfile = open(data_folder + test_path + './on_lungs/Related_hyperparameters.txt','w') 
            else:
                textfile = open(data_folder+ test_path + './on_all/Related_hyperparameters.txt','w')  
    else:
            textfile = open(data_folder + test_path + 'Related_hyperparameters.txt','w')  
    textfile.write(' Image Dimension : ' +str(image_size))
    textfile.write('\n No Labels : ' +str(output_size))

    textfile.write("\n---------------------------------")
    textfile.write("\n Averaged Test Accuracy : "+ str( test_acc.numpy()))   
                    
    textfile.write("\n---------------------------------") 
    textfile.write("\n Averaged Predictive Variance : "+ str( variance))
    textfile.write("\n---------------------------------")
    textfile.write("\n Averaged Test Dice Coefficient : "+ str( test_d1))
    
      
    textfile.write("\n---------------------------------")
    textfile.write("\n---------------------------------")
    textfile.write("\n std Test Dice Coefficient : "+ str( test_std_d1))
                      
    textfile.write("\n---------------------------------")
    if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussain_noise_std)) 
            textfile.write("\n---------------------------------")
            textfile.write("\n ('new') Averaged SNR : "+ str( new_t_snr))
            textfile.write("\n---------------------------------")
            textfile.write("\n Averaged Signal : "+ str( avr_sig))
            if noise_on =="A":
                textfile.write('\n Random Noise Applied on lungs only')
            else:
                textfile.write('\n Random Noise Applied everywhere')           
    textfile.write("\n---------------------------------")    
    textfile.close()
    return test_d1,   new_t_snr, avr_sig, path_full
 
def save_uncertainty(path,images_n, noise, where_noise ):
    print('going over function uncertainty')
    if noise == 0:
        file1 = open(path +"uncertainty_info.pkl", 'rb')
    else:
        if where_noise == "A":          
            file1 = open(path+ "uncertainty_info_on_lungs_noise_{}.pkl".format(noise,noise), 'rb')
        else:
            file1 = open(path+"uncertainty_info_noise_{}.pkl".format(noise,noise), 'rb')

    logits, sigma, truex, truey, testacc =   pickle.load(file1)
    file1.close()  
    mean_u,  class1_unc, class2_unc= save_adversarial_uncertainty(path, truex, 0, logits, truey, sigma, 0,images_n,Adversarial=False )
    return mean_u,   class1_unc, class2_unc
# noise levels
noise =  [0.01,0.02, 0.05,0.5,  0.07, 0.6]

l = len(noise)
epochs = 5
labels = 2


d1_no_noise, t_snr_no_noise,t_si_no_noise, path_no_noise =testing(epochs, labels, 0, Random_noise =False, noise_on = 'B')
mean_u_no_noise,class1_unc, class2_unc= save_uncertainty(path_no_noise,images_n=10, noise=0, where_noise='A')

no_noise= np.array([mean_u_no_noise,class1_unc, class2_unc,  d1_no_noise,t_snr_no_noise,t_si_no_noise])
print('done with no noise')
for i in noise:
    print('working with noise', i)
    d1_noise,t_snr,t_si, path =testing(epochs, labels, i,  Random_noise =True, noise_on = 'A')
    mean_u, class1_unc, class2_unc= save_uncertainty(path,images_n=10, noise=i, where_noise='A')
    
    d1_noise2,t_snr2,t_si2, path=testing(epochs, labels, i,  Random_noise =True,  noise_on = 'aa')
    mean_u2, class1_unc3, class2_unc3= save_uncertainty(path,images_n=10, noise=i, where_noise='aa')
    
