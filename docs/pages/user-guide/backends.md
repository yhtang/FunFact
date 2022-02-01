# Backends and Acceleration

FunFact is built on top of numerical backends that handle the model evaluation,
automatic differentiation, and hardware acceleration. Currently,
[JAX](https://jax.readthedocs.io/), [PyTorch](https://pytorch.org), and
[NumPy](https://numpy.org) are supported as a numerical backend. 

!!! note
    The NumPy backend only supports forward evaluation.

The available backends can be listed as:

=== "Command"
    ```py
    import funfact as ff
    ff.available_backends
    ```

=== "Result"
    ```bash
    {'jax': 'JAXBackend', 'torch': 'PyTorchBackend', 'numpy': 'NumPyBackend'}
    ```

The backend can be selected with the [`use`](../../../api/use) command
and the backend that is currently in use can be retrieved through the
`active_backend` command:

=== "NumPy"
    ```py
    from funfact import use, active_backend as ab
    use('numpy')
    ab
    ```
    ```bash
    <backend 'NumpyBackend'>
    ```

=== "JAX"
    ```py
    from funfact import use, active_backend as ab
    use('jax')
    ab
    ```
    ```bash
    <backend 'JAXBackend'>
    ```

=== "PyTorch"
    ```py
    from funfact import use, active_backend as ab
    use('torch')
    ab
    ```
    ```bash
    <backend 'PyTorchBackend'>
    ```

The FunFact active backend can be imported and used as any other NLA package:

```py
from funfact import active_backend as ab
ab.method(...)          # uses `method` from the active_backend (np, jnp, torch)
```

Besides this, there are a few other functions implemented for all backends:

```py
ab.tensor(...)          # native tensor array from NumPy array
ab.to_numpy(...)        # from native tensor to NumPy array
ab.normal(...)          # normally distributed random data
ab.uniform(...)         # uniformly distributed random data
```

## Switching backends

Backends can be switched dynamically during usage:

```py
ff.use('numpy')
a = ff.tensor(3, 2)
b = ff.tensor(2, 3)
tsrex = a @ b           # tensor expression with NumPy backend
...
ff.use('jax')
c = ff.tensor(3, 4)
d = ff.tensor(5, 6)
tsrex = a & b           # tensor expression with JAX backend
```

!!! warning
    Dynamic switching of backends does not port over previously created indices,
    tensors, and tensor expressions to the new backend. Only newly created
    expressions use the new backend.

!!! note
    Some properties of the backend can only be set once when that backend is
    initially loaded. For example, with the JAX backend, the `enable_x64` flag:
    ```py
    ff.use('jax', enable_x64=True)
    ```
    can only be set once. Running `use` a second time will not affect the
    behavior of the backend.

## Hardware acceleration

The JAX and PyTorch backends support hardware acceleration on the GPU

!!! warning
    TODO: complete this section once this functionality is implemented in `use`.




