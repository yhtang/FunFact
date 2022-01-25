# Installation

There are two options to install FunFact:

- install released versions from [PyPI](https://pypi.org/project/funfact/)
- install the latest version from source.

=== "Install from PyPI"

    ```bash
    pip install funfact
    ```

=== "Install from source"

    ```bash
    git clone https://github.com/yhtang/FunFact.git
    cd FunFact/
    python setup.py install
    ```

## Autograd backends

The default installation of FunFact installs the NumPy backend, which only
supports forward calculations. The NumPy backend doesn't support automatic differentiation
and is not able to optimize tensor expressions for methods such as [funfact.factorize](funfact.factorize).

In order to factorize tensor data by a tensor expression, two autograd backends
are provided:

=== "Use the JAX backend"

    ```bash
    pip install "funfact[jax]"
    ```

=== "Use the PyTorch backend"

    ```bash
    pip install "funfact[torch]"
    ``` 

Running the command as above will trigger installalation of the respective
packages. For more control on backend installation, please refer to:

* [JAX](https://jax.readthedocs.io/en/latest/): [installation instructions](https://github.com/google/jax#installation).

* [PyTorch](https://pytorch.org): [installation instructions](https://pytorch.org/get-started/locally/).


## Development dependencies

There are two additional sets of dependencies useful for developers:

 * To install all dependencies for generating FunFact documentation, use:
  ```bash
  pip install "funfact[docs]"
  ``` 

 * To install all dependencies for development of FunFact, use:
  ```bash
  pip install "funfact[devel]"
  ``` 
