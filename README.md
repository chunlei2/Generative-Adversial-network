# Generative-Adversial-network
Train a baseline model for CIFAR10 classification (~2 hours training time)

Train a discriminator/generator pair on CIFAR10 dataset utilizing techniques from ACGAN and Wasserstein GANs (~40-45 hours training time)

Use techniques to create synthetic images maximizing class output scores or particular features as a visualization technique to understand how a CNN is working (<1 minute)

The util.py file contains two models. The first is discriminative model. It is essencially a classifier. The second model is generative model. It receives random noise and outputs image pixels via transpose CNN.

The D_no_G file only trains the discriminator, no difference with regular classification task.

The D_with_G file train discriminator and generator together.For each iteration, you will need to update the generator network and the discriminator network separately. The generator network requires a forward/backward pass through both the generator and the discriminator. The discriminator network requires a forward pass through the generator and two forward/backward passes for the discriminator (one for real images and one for fake images). 
