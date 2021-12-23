# Grammar Reference

## Embedded Domain-Specific Language (eDSL) for Tensor Expressions

FunFact's **t**en**s**o**r** **ex**pression (**tsrex**) is an index notation system that extends the Einstein summation convention.

Below is the grammar of tsrex in BNF:

``` BNF
tsrex -> f(tsrex) |
         tsrex binary_operator tsrex |
         unary_operator tsrex |
         index_notation |
         scalar

index_notation -> tensor[indices]

indices -> indices,index |
           index
```

where the terminal symbols are

```BNF
f -> abs   |
     exp   | log   |
     sin   | cos   | tan   |
     asin  | acos  | atan  | atan2  |
     sinh  | cosh  | tanh  |
     asinh | acosh | atanh |
     erf   | erfc  |
     ...

binary_operator -> * |
                   / |
                   + |
                   - |
                   ** | 
                   % |
                   ...

unary_operator -> - |
                  ...

tensor -> identifier

index -> identifier

identifier -> ([a-zA-Z]+)(?:_([a-zA-Z\d]+))?

scalar -> real
```