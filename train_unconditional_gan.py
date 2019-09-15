# example of training an unconditional gan on the fashion mnist dataset
import os
import random
import cv2
import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D,UpSampling2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import ZeroPadding2D
from keras.initializers import RandomNormal

# define the standalone discriminator model
def define_discriminator(in_shape=(180,320,3)):
    model = Sequential()
    init = RandomNormal(stddev=0.02) 
    # downsample
    model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=in_shape,kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(64, (4,4), strides=(2,2), padding='same',kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(64, (4,4), strides=(2,2), padding='same',kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(64, (4,4), strides=(2,2), padding='same',kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(64, (4,4), strides=(2,2), padding='same',kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    #model.add(Dense(32))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

## define the standalone generator model
#def define_generator(latent_dim):
#    model = Sequential()
#    # foundation for 7x7 image
#    n_nodes = 32 * 20 * 10
#    model.add(Dense(n_nodes, input_dim=latent_dim))
#    model.add(LeakyReLU(alpha=0.2))
#    model.add(Reshape((10, 20, 32)))
#    # upsample to 14x14
#    model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
#    model.add(LeakyReLU(alpha=0.2))
#    model.add(ZeroPadding2D((1,0)))
#    # upsample to 28x28
#    model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
#    model.add(LeakyReLU(alpha=0.2))
#    # upsample to 28x28
#    model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
#    model.add(LeakyReLU(alpha=0.2))
#    model.add(ZeroPadding2D((1,0)))
#    # upsample to 28x28
#    model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
#    model.add(LeakyReLU(alpha=0.2))
#    # generate
#    model.add(Conv2D(3, (7,7), activation='tanh', padding='same'))
#    model.summary()
#    return model

# define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    init = RandomNormal(stddev=0.02) 
    # foundation for 7x7 image
    n_nodes = 32 * 20 * 10
    model.add(Dense(n_nodes, input_dim=latent_dim,kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((10, 20, 32)))
    # upsample to 14x14
    #model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')
    model.add(UpSampling2D())
    model.add(Conv2D(64, (4,4), padding='same',kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    model.add(ZeroPadding2D((1,0)))
    # upsample to 28x28
    #model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(64, (4,4), padding='same',kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 28x28
    #model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(64, (4,4), padding='same',kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    model.add(ZeroPadding2D((1,0)))
    # upsample to 28x28
    #model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(64, (4,4), padding='same',kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # generate
    model.add(Conv2D(3, (7,7), activation='tanh', padding='same',kernel_initializer=init))
    model.summary()
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    model.summary()
    return model

def load_data(dataset='shinjuku_walk/'):
  files = os.listdir(dataset)
  #random.shuffle(files)
  images=list()
  for file in files:
    if '.jpg' in file:
      img = cv2.imread(dataset + file)
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      #img = np.array(img)
      images.append(img)
  return (np.array(images),None),(None,None)
  
# load fashion mnist images
def load_real_samples():
    # load dataset
    (X, _), (_, _) = load_data()
    print (X.shape)
    # expand to 3d, e.g. add channels
    #X = expand_dims(X, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X

# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # select images
    X = dataset[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = zeros((n_samples, 1))
    return X, y

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=500, n_batch=64):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        if i != 0 and i % 10 == 0:
            # save the generator model
            g_model.save('generator_' + str(i) + '.h5')
            gan_model.save('gan_' + str(i) + '.h5')
    g_model.save('generator_' + str(n_epochs) + '.h5')
    gan_model.save('gan_' + str(i) + '.h5')

# size of the latent space
latent_dim = 256
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
print(dataset.shape)
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)
