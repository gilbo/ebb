--GPU-TEST
if not terralib.cudacompile then return end

import 'compiler.liszt'

L.default_processor = L.GPU

---------------------------
-- "Working" test cases: --
---------------------------
-- These test cases are "working" in the sense that they are not producing any errors. They
-- dont' provide as much coverage when run on the GPU, though, since the assert statement
-- has not yet been implemented on the GPU.
require 'tests.allany'
require 'tests.arith'
require 'tests.assert'
require 'tests.assigntype'
--require 'tests.const_for' -- test currently disabled
require 'tests.cross'
require 'tests.dot'
require 'tests.fieldcopyswap'
require 'tests.fieldwrite'
require 'tests.fields'
require 'tests.global'
require 'tests.kerneldecl'
require 'tests.length'
require 'tests.minmax'
require 'tests.meshsum'
require 'tests.phase'
require 'tests.shadow'
require 'tests.subsets'
require 'tests.vecindex'
require 'tests.vector'
