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
import 'ebb'
local L = require 'ebblib'

local R = L.NewRelation { name="R", size=6 }

local v = L.Constant(L.vec3f, {1, 2, 3}) 

-- We break each print statement into its own ebb function,
-- since otherwise the order of the print statements is
-- technically non-deterministic under Ebb's execution semantics

local ebb print_1 ( r : R )
    L.print(true)
end
local ebb print_2 ( r : R )
    var m = { { 1.2, 0 }, { 0.4, 1 } }
    L.print(m)
end
local ebb print_3 ( r : R )
    L.print(4)
end
local ebb print_4 ( r : R )
    L.print(2.2)
end
local ebb print_5 ( r : R )
    L.print()
end
local ebb print_6 ( r : R )
    L.print(1,2,3,4,5,false,{3.3,3.3})
end
local ebb print_7 ( r : R )
    var x = 2 + 3
    L.print(x)
end
local ebb print_8 ( r : R )
    L.print(v)
end
-- cannot rely on order of execution
--local print_stuff = ebb(r : R)
--    L.print(L.id(f))
--end

R:foreach(print_1)
R:foreach(print_2)
R:foreach(print_3)
R:foreach(print_4)
R:foreach(print_5)
R:foreach(print_6)
R:foreach(print_7)
R:foreach(print_8)
