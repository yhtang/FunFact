# Installation

There are two options to install FunFact: either from PyPI or directly from source.

## Install stable versions from [Python Package Index](https://pypi.org/project/funfact/)

```bash
pip install funfact
```

## Install the latest version from source

```bash
git clone https://github.com/yhtang/FunFact.git
cd FunFact/
python setup.py install
```

## Autograd backends

The default installation of FunFact installs the NumPy backend, which only
supports forward calculations. The NumPy backend doesn't support automatic differentiation
and is not able to optimize tensor expressions for methods such as [funfact.factorize](funfact.factorize).

In order to factorize tensor data by a tensor expression, two other backends are provides:

  * [JAX](https://jax.readthedocs.io/en/latest/) backend. Detailed 
  installation instructions for JAX can be found [here](https://github.com/google/jax#installation). Alternatively, use:
  ```bash
  pip install "funfact[jax]"
  ```

  * [PyTorch](https://pytorch.org) backend. Detailed installation instruction
  for PyTorch can be found [here](https://pytorch.org/get-started/locally/).
  Alternatively, use:
  ```bash
  pip install "funfact[torch]"
  ``` 

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
