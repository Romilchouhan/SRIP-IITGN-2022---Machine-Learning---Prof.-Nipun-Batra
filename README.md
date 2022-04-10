# SRIP IITGN 2022 Machine Learning - Prof. Nipun Batra
SRIP IITGN 2022 Screening Round Solutions

# My Learnings
JAX is an automatic differentiation toolbox which aims to bring differentiable programming in Numpy-style onto GPUs and TPUs. With its updated version of Autograd, JAX can automatically differentiate native Python and NumPy code. JAX uses [XLA](https://www.tensorflow.org/xla) to compile and run the Numpy code on accelerators, like GPUs and TPUs. Python as an interpreted programming language in slow. In order to train networks at scale we need fast compilation and parallel computing. Some precompiled CUDA kernels already provides a set of primitive instructions which can be executed massively on a NVIDIA GPU. But ideally we want to launch as few kernels as possible to reduce communication times and memory load. XLA ptimizes memory bandwith by “fusing” operations and reduces the amount of returned intermediate computations. 

<br>JAX is much more than just a GPU-backed NumPy. It also comes with a few program transformations that are useful when writing numerical code. The three main ones are:
- [jit()](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit), for speeding up your code
- [grad()](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html#jax.grad), for taking derivatives
- [vmap()](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap), for automatic vectorization or batching.

# Scope for improvements
In the solution though I tried to implement the code efficiently but still in some places the implementation can be made more effective. Here is a short list:
1. Use of pytree for model update
2. Use of more evaluation metrics for MNIST classification
3. Accuracy of the neural network can be increased by the use of some common regularisation techniques like batch-normalisation etc.
4. Variation of covariance can be made more interactive.

# Resources that helped me through:
1. [Introduction to JAX](https://www.youtube.com/watch?v=0mVmRHMaOJ4&ab_channel=GoogleCloudTech)
2. [Official github Repository for JAX](https://github.com/google/jax)
3. [A curated list of resources to learn JAX](https://github.com/n2cholas/awesome-jax)
4. [JAX Quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
5. [Training a Simple Neural Network, with tensorflow/datasets Data Loading](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/neural_network_with_tfds_data.ipynb#scrollTo=B_XlLLpcWjkA)
