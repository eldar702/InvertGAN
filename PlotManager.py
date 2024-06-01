import matplotlib.pyplot as plt

class PlotManager:
    def __init__(self):
        pass

    def plot_loss_graphs(self, epochs, gen_AB_losses, gen_BA_losses, disc_A_losses, disc_B_losses):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.plot(range(epochs), gen_AB_losses, label="Generator AB Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(range(epochs), gen_BA_losses, label="Generator BA Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(range(epochs), disc_A_losses, label="Discriminator A Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(range(epochs), disc_B_losses, label="Discriminator B Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def display_images(self, real_A, generated_B, real_B, generated_A):
        plt.figure(figsize=(5, 5))
        display_list = [real_A[0], generated_B[0], real_B[0], generated_A[0]]
        title = ['Real A', 'Generated B', 'Real B', 'Generated A']
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()
