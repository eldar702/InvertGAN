import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# Attention block for adding the self-attention mechanism
def attention_block(input_layer, filters):
    q = layers.Conv2D(filters, 1, padding='same')(input_layer) # Query
    k = layers.Conv2D(filters, 1, padding='same')(input_layer) # Key
    v = layers.Conv2D(filters, 1, padding='same')(input_layer) # Value
    qk = layers.Multiply()([q, k]) # Element-wise multiplication
    qk = layers.Activation('softmax')(qk) # Apply softmax to obtain attention
    attention = layers.Multiply()([v, qk]) # Element-wise multiplication of attention maps
    output = layers.Add()([input_layer, attention]) # Add input and attention
    return output

def define_invertible_generator(input_shape=(256, 256, 3), use_attention=False):
    input_layer = Input(shape=input_shape)

    # Initial downsampling
    x = layers.Conv2D(128, (4, 4), strides=2, padding='same')(input_layer)
    x = layers.Activation('relu')(x)

    # Apply a series of invertible blocks
    for _ in range(3):
        x = invertible_block(x, 128)

    # Optional attention mechanism
    if use_attention:
        x = attention_block(x, 128)

    # Upsampling to original dimensions
    x = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)
    output_layer = layers.Conv2D(3, (1, 1), activation='tanh')(x)

    return Model(inputs=input_layer, outputs=output_layer)

def define_discriminator(input_shape=(256, 256, 3)):
    input_layer = Input(shape=input_shape)

    # Initial layer with an even number of filters
    x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(input_layer)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Apply invertible blocks with appropriate filter counts
    x = invertible_block(x, 64)  # Stays at 64 channels
    x = invertible_block(x, 128)  # Stays at 128 channels after doubling
    x = invertible_block(x, 256)  # Stays at 256 channels after doubling

    # Flatten and use a dense layer for classification
    x = layers.Flatten()(x)
    output = layers.Dense(1)(x)

    return Model(inputs=input_layer, outputs=output)

def invertible_block(x, filters):
    assert filters % 2 == 0, "Filters must be even."
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    f_x1 = layers.Conv2D(x2.shape[-1], (1, 1), padding='same', activation='relu')(x1)
    y1 = x2 + f_x1  # Additive coupling
    g_y1 = layers.Conv2D(x1.shape[-1], (1, 1), padding='same', activation='relu')(y1)
    y2 = x1 + g_y1  # Additive coupling
    y = tf.concat([y1, y2], axis=-1)
    return y

# Initialization of models
if __name__ == "__main__":
    g_AB_invertGAN = define_invertible_generator(use_attention=True)
    g_BA_invertGAN = define_invertible_generator(use_attention=True)
    d_A_invertGAN = define_discriminator()
    d_B_invertGAN = define_discriminator()
