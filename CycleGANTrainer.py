import tensorflow as tf
import time


class CycleGANTrainer:
    def __init__(self, g_AB, g_BA, d_A, d_B, learning_rate=2e-4, beta_1=0.5, epochs=70, buffer_size=1000, batch_size=1):
        self.g_AB = g_AB
        self.g_BA = g_BA
        self.d_A = d_A
        self.d_B = d_B
        self.epochs = epochs
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_1  = beta_1
        self.setup_optimizers()

    def setup_optimizers(self ):
        all_trainable_variables = (self.g_AB.trainable_variables + self.g_BA.trainable_variables +
                                   self.d_A.trainable_variables + self.d_B.trainable_variables)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)

    @tf.function
    def train_step(self, real_A, real_B):
        with tf.GradientTape(persistent=True) as tape:
            # Generate fake images and cycle them back to the original domain
            fake_B = self.g_AB(real_A, training=True)
            cycled_A = self.g_BA(fake_B, training=True)
            fake_A = self.g_BA(real_B, training=True)
            cycled_B = self.g_AB(fake_A, training=True)

            # Generate same-domain images for identity loss
            same_A = self.g_BA(real_A, training=True)
            same_B = self.g_AB(real_B, training=True)

            # Compute the discriminator outputs for real and fake images
            disc_real_A = self.d_A(real_A, training=True)
            disc_real_B = self.d_B(real_B, training=True)
            disc_fake_A = self.d_A(fake_A, training=True)
            disc_fake_B = self.d_B(fake_B, training=True)

            # Loss calculations
            gen_AB_loss = self.generator_loss(disc_fake_B)
            gen_BA_loss = self.generator_loss(disc_fake_A)
            total_cycle_loss = self.cycle_consistency_loss(real_A, cycled_A) + self.cycle_consistency_loss(real_B,
                                                                                                           cycled_B)
            total_gen_AB_loss = gen_AB_loss + 10 * total_cycle_loss  # Lambda weight for cycle loss
            total_gen_BA_loss = gen_BA_loss + 10 * total_cycle_loss
            disc_A_loss = self.discriminator_loss(disc_real_A, disc_fake_A)
            disc_B_loss = self.discriminator_loss(disc_real_B, disc_fake_B)

            # Compute gradients
            generator_AB_gradients = tape.gradient(total_gen_AB_loss, self.g_AB.trainable_variables)
            generator_BA_gradients = tape.gradient(total_gen_BA_loss, self.g_BA.trainable_variables)
            discriminator_A_gradients = tape.gradient(disc_A_loss, self.d_A.trainable_variables)
            discriminator_B_gradients = tape.gradient(disc_B_loss, self.d_B.trainable_variables)

            # Apply gradients
            self.generator_optimizer.apply_gradients(zip(generator_AB_gradients, self.g_AB.trainable_variables))
            self.generator_optimizer.apply_gradients(zip(generator_BA_gradients, self.g_BA.trainable_variables))
            self.discriminator_optimizer.apply_gradients(
                zip(discriminator_A_gradients, self.d_A.trainable_variables))
            self.discriminator_optimizer.apply_gradients(
                zip(discriminator_B_gradients, self.d_B.trainable_variables))

        return gen_AB_loss, gen_BA_loss, disc_A_loss, disc_B_loss

    def generator_loss(self, disc_generated_output):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_generated_output,
                                                                      labels=tf.ones_like(disc_generated_output)))

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_output, labels=tf.ones_like(disc_real_output)))
        generated_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_generated_output,
                                                                                labels=tf.zeros_like(
                                                                                    disc_generated_output)))
        total_disc_loss = (real_loss + generated_loss) * 0.5
        return total_disc_loss

    def cycle_consistency_loss(self, real_image, cycled_image):
        return tf.reduce_mean(tf.abs(real_image - cycled_image))

    def train(self, train_A, train_B):
        for epoch in range(self.epochs):
            start = time.time()
            n = 0
            for real_A, real_B in tf.data.Dataset.zip((train_A, train_B)):
                self.train_step(real_A, real_B)
                if n % 10 == 0:
                    print(".", end="")
                n += 1
            print(f"\nTime taken for epoch {epoch + 1} is {time.time() - start} sec")
