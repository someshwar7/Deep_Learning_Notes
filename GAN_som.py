import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
import glob
from PIL import Image
import time 
from IPython import display
import os

# DATA PRE-PROCESSING 

(x_train_img,_),(x_test_lab,_)=tf.keras.datasets.mnist.load_data()
x_train_img=x_train_img.reshape(x_train_img.shape[0],28,28,1).astype("float64")
x_train_img=(x_train_img-127.5)/127.5 # TO SHOW THE OUTPUT IN THE INTERVAL OF -1 TO 1

# SHUFFLING THE DATA 

BUFFER=60000
BATCH_SIZE=256
train_dataset=tf.data.Dataset.from_tensor_slices(x_train_img).shuffle(BUFFER).batch(BATCH_SIZE)

# GENERATOR MODEL

def make_generator_model():
    mdl=tf.keras.Sequential()

    mdl.add(layers.Dense(7*7*256,use_bias=False,input_shape=(100,)))  # Fixed: 7*7 for proper upsampling
    mdl.add(layers.BatchNormalization())
    mdl.add(layers.LeakyReLU())

    mdl.add(layers.Reshape((7,7,256)))  
    assert mdl.output_shape == (None, 7, 7, 256)  

    mdl.add(layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding='same',use_bias=False))
    assert mdl.output_shape==(None,7,7,128) 
    mdl.add(layers.BatchNormalization())
    mdl.add(layers.LeakyReLU())

    mdl.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False))
    assert mdl.output_shape==(None,14,14,64)  #
    mdl.add(layers.BatchNormalization())
    mdl.add(layers.LeakyReLU())

    mdl.add(layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh'))  # Fixed: added tanh activation
    assert mdl.output_shape==(None,28,28,1)

    return mdl

# Calling the Generator_Model

generator=make_generator_model()
generator.summary()
noise=tf.random.normal([1,100])#error
generated_image=generator(noise,training=False)
plt.imshow(generated_image[0,:,:,0],cmap='gray')
 
# Discriminator_Model

def make_discriminator_model():
    model=tf.keras.Sequential()
    model.add(layers.Conv2D(64,(5,5),strides=(2,2),padding='same',input_shape=[28,28,1]))

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128,(5,5),strides=(2,2),padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Calling the Discriminator_model

discriminator=make_discriminator_model()
discriminator.summary()
decision=discriminator(generated_image)
print(decision)

#LOSS OF GAN
cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output , fake_output):
    real_loss=cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss=cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss=real_loss+fake_loss
    return total_loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

# optimizer for generator and discriminator

generator_optimizer=tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)

checkpoints_dir='./training_checkpoints'
checkpoints_prefix=os.path.join(checkpoints_dir,"ckpt")
checkpoints=tf.train.Checkpoint(generator_optimizers=generator_optimizer,
                                discriminator_optimizers=discriminator_optimizer
                                ,generator=generator,
                                discriminator=discriminator)

# Expermental Setup

EPOCHS=50
noise_dim=100
num_examples_to_generate=16

#we will use this seed overtime (so its easier)
#to visualize the progress in the animated GIF

seed=tf.random.normal([num_examples_to_generate,noise_dim])

# Create output image directory
os.makedirs("images", exist_ok=True)

# Training Loop
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradient_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradient_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradient_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Removed display.clear_output to retain all plots
        generate_and_save_images(generator, epoch + 1, seed) # type: ignore

        if (epoch + 1) % 15 == 0:
            checkpoints.save(file_prefix=checkpoints_prefix)

        print('Time for epoch {} is {:.2f} sec'.format(epoch + 1, time.time() - start))

    # Final image generation after all training
    generate_and_save_images(generator, epochs, seed) # type: ignore

# Generate and Save the images 

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f'images/images_at_epoch_{epoch:04d}.png')
    plt.close()

# Trainning the model
#%%time
train(train_dataset,EPOCHS)
# %%
def create_gif(image_folder='images', output_file='mnist_gan.gif'):
    # sorting the generated images that we got
    image_files = sorted(glob.glob(os.path.join(image_folder, 'images_at_epoch_*.png')))

    # Load images
    images = [Image.open(img_path) for img_path in image_files]

    # Save GIF
    if images:
        images[0].save(
            output_file,
            save_all=True,
            append_images=images[1:],
            duration=300,
            loop=0
        )
        print(f"GIF saved as: {output_file}")
    else:
        print("No images found to create GIF.")
# Saving the model

