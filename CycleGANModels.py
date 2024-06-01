import tensorflow as tf
from tensorflow.keras import layers

class Generator(tf.keras.Model):
    def __init__(self, is_attention=False, input_shape=(256, 256, 3)):
        super(Generator, self).__init__()
        self.is_attention = is_attention
        self.model = self.build_model(input_shape)

    def attention_block(self, input_layer, filters):
        q = layers.Conv2D(filters, 1, padding='same')(input_layer)  # Query
        k = layers.Conv2D(filters, 1, padding='same')(input_layer)  # Key
        v = layers.Conv2D(filters, 1, padding='same')(input_layer)  # Value
        qk = layers.Multiply()([q, k])  # Element-wise multiplication
        qk = layers.Activation('softmax')(qk)  # Apply softmax to obtain attention
        attention = layers.Multiply()([v, qk])  # Multiply value and the softmax output
        return layers.Add()([input_layer, attention])  # Sum with the input layer

    def build_model(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        if self.is_attention:
            x = self.attention_block(x, 256)

        x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        outputs = layers.Conv2D(3, 4, strides=2, padding='same', activation='tanh')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

class Discriminator(tf.keras.Model):
    def __init__(self, input_shape=(256, 256, 3)):
        super(Discriminator, self).__init__()
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
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

        x = layers.Conv2D(1, 4, padding='same')(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
