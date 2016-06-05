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


local Exports = {}
package.loaded["ebb.src.ewrap"] = Exports

local use_exp = rawget(_G,'EBB_USE_EXPERIMENTAL_SIGNAL')
if not use_exp then return Exports end

local C     = require "ebb.src.c"
local DLD   = require "ebb.lib.dld"
local Util  = require 'ebb.src.util'

-- signal to gasnet library not to try to find the shared lib or headers
rawset(_G,'GASNET_PRELOADED',true)
local gas       = require 'gasnet'
local gaswrap   = require 'gaswrap'
local distdata  = require 'distdata'

local newlist = terralib.newlist


-------------------------------------------------------------------------------
--[[            Data Accessors - Fields, Futures Declarations              ]]--
-------------------------------------------------------------------------------









