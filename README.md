This project is a port of the Liszt DSL, which was originally embedded in Scala, to the Lua/Terra system, with the following (planned) extensions:
 - implicit method support (both sparse matrix encoding and an interface to external solver libraries)
 - particles

The following extensions have already been implemented:
 - mesh data is stored on a generic OP2-like graph structure that allows us to represent mesh topology, fields and sparse matrices with the same data structure.

The original project is hosted at http://liszt.stanford.edu.

Directory structure:
compiler/ - liszt compiler implementation
examples/ - example liszt programs, and terra applications that use the liszt runtime
runtime/ - C++ liszt runtime interface used by the compiler
spec/ - contains example programs written in the Scala version of Liszt, paired with their Liszt-in-Terra equivalents.