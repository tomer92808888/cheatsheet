import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
from tensorflow.keras.preprocessing.image import *
import tensorflow as tf

class preprocessing:
    """
    Returns the labels from the directory
    """
    def get_labels(directory):
        """
        directory : (str)
        """
        data_dir = pathlib.Path(directory) # Turn the path into a Python path
        return np.array(sorted([item.name for item in data_dir.glob('*')])) # Return a list of labels from the subdirectories

    def view_random_image(target_directory, target_label):
        """
        target_directory : (str) \n 
        target_label : (str) \n
        Returns a random image from the given label from the given directory
        """
        # Setup target directory to view the images from there
        target_folder = target_directory+target_label

        # Get a random image path
        random_image = random.sample(os.listdir(target_folder), 1)

        # Read in the image and plot it using matplotlib
        img = mpimg.imread(target_folder + "/" + random_image[0])
        plt.imshow(img)
        plt.title(target_label)
        plt.axis("off")

        # Show the shape of the image
        print(f"Image shape: {img.shape}")

        return img

    def from_directory_to_batches(directory, batch_size=32, target_size=(255, 255), class_mode="binary", seed=42, augmention_setting=ImageDataGenerator(rescale=1/255.), shuffle=True):
        """
        directory : (str) \n 
        batch_size : (int) \n 
        target_size : (int,int) augmentation size which you want to give to the 2D data \n 
        class_mode : binary/ categorical \n 
        random_seed : (int) \n,
        augmentation_setting : Classes
        class DirectoryIterator: Iterator capable of reading images from a directory on disk.
        class ImageDataGenerator: Generate batches of tensor image data with real-time data augmentation.
        class Iterator: Base class for image data iterators.
        class NumpyArrayIterator: Iterator yielding data from a Numpy array.
        Functions
        apply_affine_transform(...): Applies an affine transformation specified by the parameters given.
        apply_brightness_shift(...): Performs a brightness shift.
        apply_channel_shift(...): Performs a channel shift.
        array_to_img(...): Converts a 3D Numpy array to a PIL Image instance.
        img_to_array(...): Converts a PIL Image instance to a Numpy array.
        load_img(...): Loads an image into PIL format.
        random_brightness(...): Performs a random brightness shift.
        random_channel_shift(...): Performs a random channel shift.
        random_rotation(...): Performs a random rotation of a Numpy image tensor.
        random_shear(...): Performs a random spatial shear of a Numpy image tensor.
        random_shift(...): Performs a random spatial shift of a Numpy image tensor.
        random_zoom(...): Performs a random spatial zoom of a Numpy image tensor.
        save_img(...): Saves an image stored as a Numpy array to a path or file object.
        smart_resize(...): Resize images to a target size without aspect ratio distortion. \n 

        Returns the batches of the agumented data from the given directory
        """
        return augmention_setting.flow_from_directory(directory, batch_size, target_size, class_mode, seed, shuffle)

    def load_and_prep_image(filename, target_size, batches=False):
        """
        Reads an image from filename, turns it into a tensor
        and reshapes it to (img_shape, img_shape, colour_channel).
        """
        # Read in target file (an image)
        img = tf.io.read_file(filename)

        # Decode the read file into a tensor & ensure 3 colour channels 
        # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
        img = tf.image.decode_image(img, channels=3)

        # Resize the image (to the same size our model was trained on)
        img = tf.image.resize(img, target_size)

        # Rescale the image (get all values between 0 and 1)
        img = img/255.

        # Add an extra dimension at axis 0 (Batche size)
        if batches:
            img = tf.expand_dims(img, axis=0)
        return img

    def create_tensorboard_callback(directory , model_name):
        """
        Create's the tensorboard callback in directory/mode_name provided by you 
        """
        # creating the tenosrboard callback 
        callback = tf.keras.callbacks.TensorBoard(
                log_dir=str(directory + '/' + model_name), 
            )
        return callback 
    def decode_audio(audio_binary):
        """
        decodes the audio and returns waveform 
        NOTE: the audio_binary should be binary audio file 
        use read_audio first for converting file into audio binary format 
        """
        # decoding the audio_binary file to audio and 
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)
    def read_audio(filepath):
        """
        Reading the audio wav format file and convert's it into the audio_binary format 
            filepath: (str)
        """
        # reading audio from filepaths 
        audio_binary = tf.io.read_file(filepath)
        return audio_binary 

