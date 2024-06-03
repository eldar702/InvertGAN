
# InvertGAN

## Introduction
InvertGAN is an advanced neural network model for high-fidelity image-to-image translation, improving upon the traditional CycleGAN framework by utilizing invertible networks, enhancing both the accuracy and information retention capabilities during translations.

## Repository Structure
```
InvertGAN/
│
├── CycleGANModels.py       # Contains the Generator and Discriminator models for CycleGAN.
├── CycleGAN.py             # Defines the CycleGAN network structure and training procedures.
├── CycleGANTrainer.py      # Training script for CycleGAN.
├── DatasetManager.py       # Manages the dataset loading and preprocessing.
├── ImageEvaluator.py       # Evaluates the model performance using PSNR and SSIM.
├── InvertGAN.py            # Main file for the InvertGAN model, including network definitions.
├── train_invertgan.py      # Training procedures for InvertGAN.
├── main.py                 # Entry point for running the models.
└── PlotManager.py          # Utility for plotting training results and images.
```

## Installation
Ensure you have Python 3.8 or higher installed, then install the required packages:
```
pip install tensorflow
pip install matplotlib
pip install scikit-image
pip install numpy
pip install pillow
```

## Dataset
The model is trained on a dataset of 2000 paintings by Bela Czobel and Camille Pissarro, focusing on a variety of artistic styles and themes to demonstrate the capabilities of InvertGAN in preserving artistic integrity during style translation.

## Results
InvertGAN has been shown to maintain higher detail fidelity and fewer artifacts compared to traditional CycleGAN, particularly in complex areas such as intricate textures and sharp color transitions. Performance metrics such as PSNR and SSIM are used to quantitatively assess the model's performance.
![alt text](https://github.com/eldar702/InvertGAN/tree/main/generated%20photos/Picture1.png?raw=true)

## Contributing
Contributions are welcome. Please fork the repository and submit a pull request.





## Contact
For any queries or further information, please contact [Your Email].

---

Make sure to fill in or adjust any placeholders, such as the `License` or `Contact` sections, based on your actual project details or preferences.
