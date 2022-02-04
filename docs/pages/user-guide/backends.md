# Backends and Acceleration

FunFact delegates numerical linear algebra operations, automatic differentiation,
and hardware acceleration to external linear algebra packages that conforms to the
NumPy API. Currently supported backends include [JAX](../../../api/backend/_jax),
[PyTorch](../../../api/backend/_torch), and [NumPy](../../../api/backend/_numpy).

!!! note
    The NumPy backend only supports forward evaluation.

The list of available backends can be query by:

=== "Command"
    ```py
    import funfact as ff
    ff.available_backends
    ```

=== "Result"
    ```bash
    ['jax', 'torch', 'numpy']
    ```

The backend can be selected with the [`use`](../../../api/use) method.
The backend that is currently in use can be retrieved through the
`active_backend` method:

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

The active backend can be imported and used as if it is the underlyinng NLA package:

```py
from funfact import active_backend as ab
ab.eye(...)
ab.zeros(...)
# uses any method already defined by the underlying package (np, jnp, torch)
ab.*method*(...)
```

Besides this, a FunFact backend implements a few additional methods such as:

```py
ab.tensor(...)          # create native tensor from array-like data
ab.to_numpy(...)        # convert native tensor to NumPy array
ab.normal(...)          # normally distributed random data
ab.uniform(...)         # uniformly distributed random data
```

## Switching backends

Backends maybe be switched dynamically during the lifetime of a process:

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
    Dynamic switching of the backends will not automatically port the data
    in an existing tensor expression/model to the new backend.

!!! note
    Some properties of the backend can only be set once when that backend is
    loaded for the first time in a process. For example, with the JAX backend,
    the `enable_x64` flag:
    ```py
    ff.use('jax', enable_x64=True)
    ```
    can only be set once. Running `use` a second time will not affect this
    behavior.

## Hardware acceleration

The JAX and PyTorch backends support hardware acceleration on the GPU

!!! warning
    TODO: complete this section once this functionality is implemented in `use`.




