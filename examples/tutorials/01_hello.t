-- The MIT License (MIT)
-- 
-- Copyright (c) 2015 Stanford University.
-- All rights reserved.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included
-- in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------

import 'ebb'
-- The program starts by importing the Ebb language.  Every file you write
-- that includes Ebb code (rather than pure Lua) needs to begin with
-- this command.


local L = require 'ebblib'
-- In addition to importing the language, we probably also want to `require`
-- Ebb's standard library.  `require` is the standard Lua mechanism for 
-- pulling in modules, similar to C's `#include`, or Java/Python's `import`.


local GridLibrary = require 'ebb.domains.grid'
-- In addition to the Ebb standard library, we usually `require` some number
-- of other support libraries.  In particular, we'll usually want to require
-- at least one geometric domain library.  Ebb provides a set of
-- default domain libraries available at 'ebb.domains.xxxx';
-- Here we use the grid library.


local grid = GridLibrary.NewGrid2d {
  size          = {2,2},
  origin        = {0,0},
  width         = {2,2},
}
-- Using the grid library, we can create a new domain.  Here we're telling
-- the library that we want a 2x2 (`size`) 2d grid, with its origin
-- at (0,0) and with grid width 2 in each dimension.


local ebb printsum( c : grid.cells )
  L.print(21 + 21)
end
-- After creating a domain, we define computations over that domain
-- using Ebb functions.  Here, we define a function `printsum` which
-- takes a cell of the grid as its argument.  We sometimes call functions
-- that take domain elements as their only argument _kernels_.  These
-- kernels represent data parallel computations over the geometric domain.

grid.cells:foreach(printsum)
-- Finally, we invoke this function for each cell in the grid.
-- Since there are 4 cells, this will print out the sum 4 times.
