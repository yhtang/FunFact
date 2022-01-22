# Grammar Reference

!!! note
    The information on this page is mostly relevant to advanced users and developers.


## Embedded Domain-Specific Language (eDSL) for Tensor Expressions

As we have seen [earlier](../tsrex), a FunFact **t**en**s**o**r** **ex**pressions (**tsrex**) can be expressed using a hybrid of:

- index notations that extends the Einstein summation convention, and
- NumPy-style operations.

This expression system essentially implements a domain-specific language (eDSL) embedded in Python. The formal grammar for this eDSL is:

```BNF
tsrex -> f(tsrex) |
         tsrex binary_operator tsrex |
         unary_operator tsrex |
         index_notation |
         tensor |
         literal

index_notation -> tsrex[indices]

indices -> index |
           indices,  index |
           indices, ~index |
           indices, *index

f -> abs   |
     exp   | log   |
     sin   | cos   | tan   |
     asin  | acos  | atan  | atan2  |
     sinh  | cosh  | tanh  |
     asinh | acosh | atanh |
     erf   | erfc  |
     ...

binary_operator -> *  |
                   /  |
                   +  |
                   -  |
                   ** |
                   &  |
                   ...

unary_operator -> - |
                  ...

tensor -> identifier

index -> identifier

identifier -> ([a-zA-Z]+)(?:_([a-zA-Z\d]+))?

literal -> number
```