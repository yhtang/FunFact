# Grammar Reference

!!! note
    The information on this page is mostly relevant to advanced users and developers.


## Embedded Domain-Specific Language (eDSL) for Tensor Expressions

As we have seen [earlier](../tsrex), a FunFact **t**en**s**o**r** **ex**pressions (**tsrex**) can be expressed using a hybrid of:

- index notations that extends the Einstein summation convention, and
- NumPy-style operations.

This expression system essentially implements a domain-specific language (eDSL) embedded in Python. The formal grammar for this eDSL is:

=== "Rule"

    - An elementwise function evaluation of a tensor expression yields a new tensor expression.
    - Binary operations between two tensor expressions yields a new tensor expression.
    - Unary operations on a tensor expression yields a new tensor expression.
    - An index notation is by itself a tensor expression.
    - A tensor is by itself a tensor expression.
    - A literal value is by itself a tensor expression.

=== "BNF"

    ```
    tsrex -> f(tsrex) |
             tsrex binary_operator tsrex |
             unary_operator tsrex |
             index_notation |
             tensor |
             literal
    ```

---

=== "Rule"

    A tensor expression, regardless of its complexity, can be indexed by an
    index set whose size is consistent with its dimensionality.

=== "BNF"

    ```
    index_notation -> tsrex[indices]
    ```

---

=== "Rule"

    A valid index set consists of zero or more index variables, each of which
    can be optionally decorated with the `~` and `*` modifier.

=== "BNF"

    ```
    indices -> |
               index |
               indices,  index |
               indices, ~index |
               indices, *index
    ```

---

=== "Rule"

    Most common math routines in NumPy can be used as elementwise functions.

=== "BNF"

    ```
    f -> abs   |
         exp   | log   |
         sin   | cos   | tan   |
         asin  | acos  | atan  | atan2  |
         sinh  | cosh  | tanh  |
         asinh | acosh | atanh |
         erf   | erfc  |
         ...
    ```

---

=== "Rule"

    Valid binary operators are multiplication, division, addition, subtraction,
    exponentiation, Kronecker product, and matrix multiplication.

=== "BNF"

    ```
    binary_operator -> *  |
                       /  |
                       +  |
                       -  |
                       ** |
                       &  |
                       @
    ```

---

=== "Rule"

    The only unary operator currently implemented is negation.

=== "BNF"

    ```
    unary_operator -> -
    ```

---

=== "Rule"

    Tensors and indices can be named by a alphanumeric identifier with an
    optional numeric subscript.

=== "BNF"

    ```
    tensor -> identifier
    index -> identifier
    identifier -> ([a-zA-Z]+)(?:_([a-zA-Z\d]+))?
    ```

---

=== "Rule"

    A literal is a real or complex scalar whose value is known at the time of
    creation of a tensor expression.

=== "BNF"

    ```
    literal -> number
    ```