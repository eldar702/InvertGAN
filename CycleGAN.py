import tensorflow as tf
from tensorflow.keras import layers
from IPython.display import clear_output
import time


# Generator Class
class Generator(tf.keras.Model):
    def __init__(self, is_attention=False, input_shape=(256, 256, 3)):
        super(Generator, self).__init__()
        self.is_attention = is_attention
        self.input_shape = input_shape
        self.build_model()

    def attention_block(self, input_layer, filters):
        q = layers.Conv2D(filters, 1, padding='same')(input_layer)  # Query
        k = layers.Conv2D(filters, 1, padding='same')(input_layer)  # Key
        v = layers.Conv2D(filters, 1, padding='same')(input_layer)  # Value
        qk = layers.Multiply()([q, k])  # Element-wise multiplication
        qk = layers.Activation('softmax')(qk)  # Apply softmax to obtain attention
        attention = layers.Multiply()([v, qk])  # Element-wise multiplication
        output = layers.Add()([input_layer, attention])  # Add input and attention
        return output

    def build_model(self):
        input_layer = layers.Input(shape=self.input_shape)
        # Encoder
        x = layers.Conv2D(64, 4, strides=2, padding='same')(input_layer)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # Bottleneck with optional attention
        x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        if self.is_attention:
            x = self.attention_block(x, 512)

        # Decoder
        x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        output_layer = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(x)

        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    def call(self, inputs):
        return self.model(inputs)


# Discriminator Class
class Discriminator(tf.keras.Model):
    def __init__(self, input_shape=(256, 256, 3)):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.build_model()

    def build_model(self):
        input_layer = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(64, 4, strides=2, padding='same')(input_layer)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(1, 4, padding='same')(x)  # Output layer
        self.model = tf.keras.Model(inputs=input_layer, outputs=x)

    def call(self, inputs):
        return self.model(inputs)
