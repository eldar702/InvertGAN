import tensorflow as tf
from InvertGAN import Generator, Discriminator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from IPython.display import clear_output
import time



class InvertGANTrainer:
    def __init__(self, generator_AB, generator_BA, discriminator_A, discriminator_B, learning_rate=2e-4, beta_1=0.5,
                 buffer_size=1000, batch_size=1, epochs=10):
        self.g_AB = generator_AB
        self.g_BA = generator_BA
        self.d_A = discriminator_A
        self.d_B = discriminator_B
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.setup_optimizers()

    def setup_optimizers(self):
        all_trainable_variables = (self.g_AB.trainable_variables + self.g_BA.trainable_variables +
                                   self.d_A.trainable_variables + self.d_B.trainable_variables)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)

        # Optionally building the optimizers if needed, though it's not typically required as TensorFlow handles this automatically during the first train step.
        self.generator_optimizer.build(all_trainable_variables)
        self.discriminator_optimizer.build(all_trainable_variables)
    def discriminator_loss(self, real, generated):
        real_loss = self.loss_object(tf.ones_like(real), real)
        generated_loss = self.loss_object(tf.zeros_like(generated), generated)
        total_disc_loss = (real_loss + generated_loss) * 0.5
        return total_disc_loss

    def generator_loss(self, generated):
        return self.loss_object(tf.ones_like(generated), generated)

    def train_step(self, real_A, real_B):
        with tf.GradientTape(persistent=True) as tape:
            fake_B = self.g_AB(real_A, training=True)
            cycled_A = self.g_BA(fake_B, training=True)
            fake_A = self.g_BA(real_B, training=True)
            cycled_B = self.g_AB(fake_A, training=True)

            disc_real_A = self.d_A(real_A, training=True)
            disc_real_B = self.d_B(real_B, training=True)
            disc_fake_A = self.d_A(fake_A, training=True)
            disc_fake_B = self.d_B(fake_B, training=True)

            gen_AB_loss = self.generator_loss(disc_fake_B)
            gen_BA_loss = self.generator_loss(disc_fake_A)
            disc_A_loss = self.discriminator_loss(disc_real_A, disc_fake_A)
            disc_B_loss = self.discriminator_loss(disc_real_B, disc_fake_B)

        gradients_of_generators = tape.gradient([gen_AB_loss, gen_BA_loss], [self.g_AB.trainable_variables, self.g_BA.trainable_variables])
        gradients_of_discriminators = tape.gradient([disc_A_loss, disc_B_loss], [self.d_A.trainable_variables, self.d_B.trainable_variables])

        self.generator_optimizer.apply_gradients(zip(gradients_of_generators[0], self.g_AB.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(gradients_of_generators[1], self.g_BA.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminators[0], self.d_A.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminators[1], self.d_B.trainable_variables))

    def train(self, train_A, train_B):
        for epoch in range(self.epochs):
            start = time.time()
            for image_A, image_B in tf.data.Dataset.zip((train_A, train_B)):
                self.train_step(image_A, image_B)
            clear_output(wait=True)
            print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')

