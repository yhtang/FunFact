# Tensor Factorization

A FunFact [tensor expression](../tsrex) can be used to *factorize* a target
data tensor with the `factorize` method. The most basic usage is as follows:
```py
import funfact as ff
# ... create tsrex, load target
fac = ff.factorize(tsrex, target)
```
The result `fac` is a [factorization model](../eval) based on the input `tsrex`.
The solution is found by minimizing the mean squared error (MSE) loss function
between the data and factorized tensor using stochastic gradient descent.

!!! note
    The `target` data tensor has to be a native tensor type of the [active 
    backend](../backends) or a NumPy array. Furthermore, the `tsrex` expression 
    and `target` have to be of the same shape.

The `factorize` algorithm can be adjusted and fine-tuned by the user in order 
to obtain the desired performance for each application. By default, the 
iteration will run for `max_steps=10000` steps or until convergence is reached 
with tolerance `tol=1e-6`, whichever occurs first. 

## Vectorization for multi-instance learning

Out of the box, the `factorize` method creates one random initialization of the
factorization model as starting point for the optimization process. For more
complicated tensor factorizations, the chances of success can be drastically
improved by optimizing multiple random initializations simultaneously.
This process is called *vectorization* in FunFact and the `factorize` algorithm
can be used on a vectorized model  by specifying the `nvec` keyword integer
argument.

```py
fac = ff.factorize(tsrex, target, nvec=64)
```

The code above will run 64 randomly initialized models in parallel until
`max_steps` is reached or one of the model instances reaches the convergences
threshold. At that point, the best factorization model in terms of loss is
returned.

The behavior of a vectorized iteration can be adjusted by the following keyword
arguments for the `factorize` algorithm:

- `stop_by=...` indicates when the iteration for a vectorized factorization
stops:
    * `first`: the iteration stops as soon as one instance satisfies the
    convergence criterion.
    * `int n: nvec >= n >= 1`: the iteration stops as soon as `n` instances
    satisfy the convergence criterion.
    * `None`: always run the iteration until `max_steps`.
The default is `first`.

- `returns=...` specifies what output is returned:
    * `best`: only the best instance is returned as a factorization model
    * `int n: nvec >= n >= 1`: the `n` best instances are returned as a list of
    factorization models.
    * `all`: all the instances are returned as a single *vectorized*
    factorization model.
The default is `best`.

- `apppend=...` is a boolean flag that indicates if the vectorization dimension
is appended (`True`) or prepended (`False`) to the leaf tensors. This does not
effect convergence, but can improve the performance dependent on the order the
data is stored in memory. The default value is `False`.

An example use case is:

```py
fac = ff.factorize(tsrex, target, 
                   nvec=64, 
                   append=False, 
                   stop_by=None, 
                   returns=8
)
```
## Fine-tuning 

The `factorize` algorithm can be further tailored to a specific application and
data set. To following keyword arguments are available to users to achieve this:

- `tol` (float): convergence threshold imposed on loss function. Default: 
`1e-6`.
- `max_steps` (int): maximum number of iterations. Default: `10000`.
- `checkpoint_freq` (int): Frequency that convergence is checked. Default: every
`50` iterations.
- `dtype` (ab.dtype): Numerical datatype. Default: `None` which uses the same
type as `target`.

### Optimizer

The optimizer that is used can be changed and its (hyper)parameters, such as the
learning rate `lr`, can be modified through the FunFact
[Optimizer API](../../../api/optim/). All keyword arguments that are input
to `factorize` are passed to the optimizer.
FunFact currently offers native support for two optimizers and allows for
a custom optimizer:

- `optimizer`: optimization algorithm:
    * `'Adam'`: Adam optimizer.
    * `'RMSprop'`: RMSprop optimizer.
    * `callable`: user provided optimizer that implements the [Optimizer 
    API](../../../api/optim/).

### Loss

The loss function used in the optimization process can be changed as well as its
(hyper) parameters. This adheres to the FunFact [Loss API](../../../api/loss/).
All keyword arguments that are input to `factorize` are passed to the loss
function.
FunFact currently offers native support for three loss functions and allows
for a custom loss function:

- `loss`: loss function:
    * `'mse_loss'`: Mean-Squared Error (L2) loss.
    * `'l1_loss'`: L1 loss.
    * `'kldiv_loss'`: KL Divergence loss.
    * `callable`: user provided optimizer that implements the [Loss 
    API](../../../api/loss/).

The loss function can be further adjusted by the `penalty_weight` argument which
is the scalar factor applied to all the penalty terms specified on the leaf
tensors. If `penalty_weight=0.0`, it is not taken into account in the loss
function. The default value is `penalty_weight=1.0`.