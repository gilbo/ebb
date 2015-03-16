
# Tensor Indexing Feature Proposal

alternate names? Inline Indexing?


## Assorted Motivations

- Pat and Zach were interested in including some kind of APL-style syntax in Liszt a while back.
- Liszt now has small vector and matrix types; programmers would benefit from some kind of syntax to make it easier to write expressions involving these kinds of data.
- Fine-grained parallelism:  When a kernel has a large working set, we would like to have the option of mapping each row of the relation we're mapping over onto multiple SIMD lanes of the GPU.  For instance, in the VEGA FEM code, we found that executing with a block size of 16 (and simply letting half of the lanes go unused) produced speedups, presumably because of reduced memory pressure.  To recover some of this lost compute we need to identify intra-kernel parallelism.


## Some Inspirations

- Tensor notaions from math, especially Einstein notation
- Big sum notations and some of Guy Steele's ideas about Chapel syntax.


## A Motivating Example: Lots of different products

Consider the following variables
```
var A, B, C : L.mat3d
var x, y, z : L.vec3d
var d : L.double
```

From this data, we can define a number of different products we might want to compute

- dot product `d = x*y`
- point-wise product `z = x*y`
- outer product `A = x*y`
- dot product `d = A*B`
- point-wise product `C = A*B`
- matrix-matrix product `C = A*B`
- matrix-vector product `y = A*x`
- matrix columns scaled by vector `B = A*x`

How might we handle this situation?

1) declare one meaning of the multiplication operator correct by fiat. (e.g. point-wise multiplication)  This is unambiguous, but prevents the programmer from directly expressing many of these products.  The products can be emulated with little `for` loops, but doing so is verbose and tends to make compiler analysis more difficult.

2) use APL like adverbs.  This is a fairly elegant solution but tends to be hard for programmers to understand.  Here is how we might encode the above products in a J-like syntax mashed together with more conventional syntax

- dot product `d = +/ (x * y)`
- point-wise product `z = x*y`
- outer product `A = x ,: y`
- dot product `d = +/ +/ (A * B)`
- point-wise product `d = A * B`
- matrix-matrix product `C = A (/+ . *) B`
- matrix-vector product `y = A (/+ . *) B`
- matrix columns scaled by vector (not sure how to do this)

While these expressions are concise, it can be relatively difficult to understand what exactly the operators are doing or can possibly do.  It is also very difficult to explain this syntax to someone unfamiliar with it.  These are known problems

I want to propose another alternative that is clearer than APL.  Here is how we would write the products discussed above in the proposed syntax

- dot product `d = +[i] x[i] * y[i]`
- point-wise product `z = _[i] x[i] * y[i]`
- outer product `A = _[i,j] x[i] * y[j]`
- dot product `d = +[i,j] A[i,j] * B[i,j]`
- point-wise product `C = _[i,j] A[i,j] * B[i,j]`
- matrix-matrix product `C = _[i,j] +[k] A[i,k]*B[k,j]`
- matrix-vector product `y = _[i] +[j] A[i,j]*x[j]`
- matrix columns scaled by vector `B = _[i,j] A[i,j]*x[j]`

I claim that this syntax is more straightforward for a novice to understand at a small cost of verboseness.  As I'll discuss later, it can also be translated into very natural parallel primitives.



# Syntax, Semantics, and Implementation

## Syntax and Intuitive Semantics

Tensor indexing can be introduced via one new piece of concrete syntax, that I'll call a big-operator.

```
operator[name1,...] expression
```

Big operators `op[]` will be understood as low-precedence such that they capture as large of an expression as possible to its right.

Big operators come in two flavors

- reduction/fold operators: e.g. `+[i]`.  This is basically a big sigma summation-style operation.  It says, compute this expression for every possible value of `i` and then sum over the results.

- the map operator: e.g. `_[i]`.  This says to compute the expression for every possible value of `i` and then to generate a vector of resulting values.

Notice that big operators also function as a variable name *binding site*.  That is, in `_[i,j]` the variable names `i` and `j` are being introduced.  In `+[i]` the variable name `i` is being introduced.  The scope of these variables is precisely the expression to the right.

Big operators leave one important piece of information implicit though; the range of each introduced variable.  We rely on typechecking to infer this bit of information



## Typechecking

When a big operator index variable is introduced we don't know what range of values it should assume.  However, we assume that it will naturally range over the indices of some matrix or vector, from which we can infer the correct range.  If more than one correct range is inferred, then we report an error.

Optional:  We may want to allow programmers to explicitly annotate the range.  For instance, we could write something like `+[i:4]` to say `i` ranges over the values `0,1,2,3`, but I'm not convinced this is that desirable a feature, and it is a little bit ugly.



## Implementation

We can trivially replace each big operator with a loop or unrolled loop.



## Optimization

The map operator tends to have fairly loose distributivity laws, which allows us the choice of commuting it with most other big operators.

If we assume certain distributivity laws, then we can also do loop hoisting.  e.g. we can transform the first expression below into the second, which eliminates about 6 multiplications from the arithmetic

```
d = +[i,j] +[i] * A[i,j] * y[j]
d = +[i] +[i] * (_[k] +[j] A[k,j] * y[j])[i]
```



# Some other cool examples of things you can do with big operators

```
-- slicing out the first column of a matrix
x = _[i] A[i,0]

-- transposing a matrix
B = _[i,j] A[j,i]

-- Summing a matrix into a single row
x = _[j] +[i] A[i,j]
```



# Major limitation

You can't really swizzle a vector with this feature, which makes it hard to implement cross products.















