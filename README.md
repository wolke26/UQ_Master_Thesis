# UQ_Master_Thesis

In this project you can find some ingredients to sample Gaussian and Non-Gaussian Random fields (Beta, Gamma, Uniform, Logarithmic).
This code was used for my Master Thesis to analyse Uncertainty Quantification of Arterial Wall Simulations with Neural Networks which was conducted at Graz University of Technology in 2020-2021. 
If you are interested in my thesis or on different sampling methods of Random Fields (Gaussian and Non-Gaussian), please feel free to reach out to me anytime.

Moreover, the results of this work will be published in a paper (link to hopefully follow) for which some sample snippets might be of use as well.

The Deep Bayesian Encoder Decoder Network is based on the work "Bayesian deep convolutional encoder–decoder networks for surrogate modeling and uncertainty quantification" done by Zhu & Zabaras 2018 and can be found here https://github.com/cics-nd/cnn-surrogate , as well as the paper "http://www.sciencedirect.com/science/article/pii/S0021999118302341".

For personal use an adapted clone of their work is uploaded to this folder as well. If you need access to the underlying data, please contact me via mail at wolkerstorfer@biomed.ee.ethz.ch


Little overview of the Project, this folder is structures as follows:
- Random Field generation with Spectral Method
- Bayesian Encoder Decoder


In the file RandomFields one can run the code and play around with some hyperparameters for Random Field generation.
The Folder Deep_Bayesian_Encoder_Decoder contains afolder named "Test Data" with which the network was tested with. The training Data can be accessed upon reasonable request to the authors.

The folder /Deep_Bayesian_Encoder_Decoder/models contains the training models of the BNN. "bayes_nn.py" contains the Bayesian Neural Network, "dense_ed.py" the Encoder-Decoder and "svgd.py" contains the Stein-Variational-Gradient-Descent algorithm.

In the subfolder /Deep_Bayesian_Encoder_Decoder/utils one cna find "lhs.py" for latin hypercude sampling when performing the Uncertainty Quantification, "load_data.py" is used to load the Data for training/testing and evaluation, "misc.py" contains several useful functions, e.g. automatic folder generation for evaluation and saving models, "plot.py" is used to automatically vizualize the output of the training, e.g. when performing the Uncertainty Quantification the results are plotted with functions defined in there.

The file/Deep_Bayesian_Encoder_Decoder/args.py contains all arguments of the network including batch-size, size for down and upsampling, debth of the network and directory names for saving the output.

The file /Deep_Bayesian_Encoder_Decoder/post_processing.py performs the Uncertainty Quantification and plots a prediction of the network at a certain location, as well as performing and plotting the uncertainty distribution at a specific location and the reliability diagram of the network.
Therefore the file /Deep_Bayesian_Encoder_Decoder/uq.py is utilized.

To run the training the file use /Deep_Bayesian_Encoder_Decoder/train_svgd.py.

