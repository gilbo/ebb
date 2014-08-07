import 'compiler.liszt'

L.default_processor = L.GPU

---------------------------
-- "Working" test cases: --
---------------------------
-- These test cases are "working" in the sense that they are not producing any errors. They
-- dont' provide as much coverage when run on the GPU, though, since the assert statement
-- has not yet been implemented on the GPU.
terralib.require 'tests.allany'
terralib.require 'tests.arith'
terralib.require 'tests.assert'
terralib.require 'tests.assigntype'
--terralib.require 'tests.const_for' -- test currently disabled
terralib.require 'tests.cross'
terralib.require 'tests.dot'
terralib.require 'tests.fieldcopyswap'
terralib.require 'tests.fieldreduce'
terralib.require 'tests.fieldwrite'
terralib.require 'tests.fields'
terralib.require 'tests.global'
terralib.require 'tests.kerneldecl'
terralib.require 'tests.length'
terralib.require 'tests.meshsum'
terralib.require 'tests.phase'
terralib.require 'tests.shadow'
terralib.require 'tests.subsets'


-- Prints stuff...
--terralib.require 'tests.functions'
--terralib.require 'tests.luabuiltins'

------------------------
-- Broken test cases: --
------------------------
--terralib.require 'tests.minmax'      -- reduction support for max/double
--terralib.require 'tests.nest'        -- cuda compile error
--terralib.require 'tests.terrabridge' -- cudacompile error
--terralib.require 'tests.vecindex'    -- reduction support +/double
--terralib.require 'tests.vector'      -- reduction support +/double


