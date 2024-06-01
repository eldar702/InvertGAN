import tf as tf

from CycleGAN import Discriminator, Generator
from CycleGANTrainer import CycleGANTrainer
from DatasetManager import DatasetManager
from ImageEvaluator import ImageEvaluator
from InvertGAN import define_invertible_generator, define_discriminator
from train_invertgan import InvertGANTrainer

if __name__ == "__main__":
    ################              Dataset creation              ################
    manager = DatasetManager()
    print("Loading bela-czobel dataset...")
    bela_dataset = manager.load_dataset("bela-czobel")
    print("Loading camille-pissarro dataset...")
    camille_dataset = manager.load_dataset("camille-pissarro")
    bela_train, bela_test, camille_train, camille_test = manager.split_datasets(bela_dataset, camille_dataset)

    test_A = tf.data.Dataset.from_tensor_slices(bela_test).take(50).batch(1)
    test_B = tf.data.Dataset.from_tensor_slices(camille_test).take(50).batch(1)

    ################              InvertGAN              ################
    g_AB_invertGAN = define_invertible_generator()
    g_BA_invertGAN = define_invertible_generator()
    d_A_invertGAN = Discriminator()
    d_B_invertGAN = Discriminator()
    trainer = InvertGANTrainer(g_AB_invertGAN, g_BA_invertGAN, d_A_invertGAN, d_B_invertGAN)
    trainer.train(bela_train, camille_train)
    evaluator = ImageEvaluator()



    print("***** invertGAN Images *****")
    for real_A, real_B in zip(test_A, test_B):
        generated_B = g_AB_invertGAN(real_A, training=False)
        generated_A = g_BA_invertGAN(real_B, training=False)
        evaluator.display_images(real_A.numpy(), generated_B.numpy(), real_B.numpy(), generated_A.numpy())


    ################              CycleGAN              ################
    # Instantiate CycleGAN components
    g_AB = Generator(is_attention=False)
    g_BA = Generator(is_attention=False)
    d_A = Discriminator()
    d_B = Discriminator()

    trainer = CycleGANTrainer(g_AB, g_BA, d_A, d_B)
    trainer.train(bela_train, camille_train)  # Train the model with appropriate datasets
    # Testing setup
    evaluator = ImageEvaluator()
    for real_A, real_B in zip(test_A, test_B):
        generated_B = g_AB(bela_test, training=False)
        generated_A = g_BA(camille_test, training=False)
        evaluator.display_images(real_A.numpy(), generated_B.numpy(), real_B.numpy(), generated_A.numpy())



