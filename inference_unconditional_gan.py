# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
import numpy as np
import os
import scipy.misc
import cv2



# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# create and save a plot of generated images (reversed grayscale)
def show_plot(examples, n):
    # plot images
    fig = pyplot.figure()
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        #rgb = scipy.misc.toimage(examples[i, :, :, 0])
        rgb = examples[i, :, :,:]
        rgb = scipy.misc.toimage(rgb)
        #print(np.max(rgb),np.min(rgb))
        pyplot.imshow(rgb)
    pyplot.show()
    fig.savefig('plot.png')

# load model
#model = load_model('saved/generator_450.h5')
model = load_model('saved/generator_120_big.h5')
# generate images
#latent_points = generate_latent_points(128,128)
latent_points = generate_latent_points(192,192)
# generate images
X = model.predict(latent_points)
print(X.shape)
# plot the result
show_plot(X, 4)
