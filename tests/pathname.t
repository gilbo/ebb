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

local test = require('tests.test')

local PN = require('ebb.lib.pathname')
local Pathname = PN.Pathname

local testdir = Pathname.new('pathname_testdir')

-- check that our constant paths are actually absolute / root
test.eq(Pathname.root:is_absolute(), true)
test.eq(Pathname.root:is_root(), true)
test.eq(Pathname.pwd:is_absolute(), true)
test.eq(Pathname.ebb_root:is_absolute(), true)

test.eq(Pathname.root:basename(), '')

-- exercise absolute and relative path tests
test.eq(testdir:is_relative(), true)
test.eq(testdir:is_absolute(), false)
test.eq(testdir:abspath():is_relative(), false)
test.eq(testdir:abspath():is_absolute(), true)
test.eq(testdir:is_root(), false)

-- check the script path here
test.eq(PN.scriptdir():basename(), 'tests')

-- rebuild a clean testdir
if testdir:exists() then
  os.execute('rm -r pathname_testdir')
end
assert(not testdir:exists())
test.eq(testdir:mkdir(), true) -- should succeed
-- now the directory should exist, regardless of whether we use
-- relative or absolute paths to reference it
test.eq(testdir:exists(), true)
test.eq(testdir:abspath():exists(), true)

-- test concatenation overloading
local foodir = testdir .. 'foo'
local redundant = '.' .. foodir

-- composing strings with pathnames should always promote to a
-- pathname regardless of whether the composition was on the left or right
test.eq(PN.is_pathname(redundant), true)

test.eq(redundant:tostring(), './pathname_testdir/foo')
-- the clean version should be collapsed back down again
test.eq(redundant:cleanpath():tostring(), foodir:tostring())

-- test parent function (will clean string)
test.eq(foodir:parent():tostring(), testdir:tostring())

-- do a basic test some of the path decomposition
test.eq(foodir:dirpath():tostring(), testdir:tostring())
test.eq(foodir:basename(), 'foo')
test.eq(foodir:extname(), '')
test.eq(foodir:dirname(), 'pathname_testdir')
test.eq(foodir:absdirname(), foodir:absdirpath():tostring())
test.eq(foodir:absdirname(), PN.pwd_str()..'/pathname_testdir')

-- foo should not exist yet, let's test that and create it
test.eq(foodir:exists(), false)
test.eq(foodir:direxists(), true)
test.eq(foodir:is_directory(), false) -- it can't be anything til it exists
test.eq(foodir:is_file(), false)

-- actually create dir foo
test.eq(foodir:mkdir(), true)
-- test that trying to create again will fail
test.eq(foodir:mkdir(), false)
-- show that foo exists how we expect
test.eq(foodir:exists(), true)
test.eq(foodir:is_directory(), true)
test.eq(foodir:is_file(), false)


-- now construct an extra long path and confirm that mkdir
-- fails and that no directories were constructed
local longpath = testdir .. 'something/or_another'
test.eq(longpath:mkdir(), false)
test.eq(longpath:direxists(), false)


-- create some non-directory files
os.execute('touch '..testdir:tostring()..'/.hidden')
os.execute('touch '..testdir:tostring()..'/whitespace\\ posix-portable\\ name.txt')
os.execute('touch '..testdir:tostring()..'/noext.t.')
os.execute('touch '..testdir:tostring()..'/boring.t')

local hiddenpath = testdir..'.hidden'
local noextpath  = testdir..'noext.t.'
local boringpath = testdir..'boring.t'
test.eq(noextpath:extname(), '')
test.eq(hiddenpath:extname(), '')
test.eq(boringpath:extname(), 't')

local dir_contents = {
  ['foo'     ]  = true,
  ['noext.t.']  = true,
  ['boring.t']  = true,
  ['whitespace posix-portable name.txt'] = true
}
local dir_w_hidden_contents = {
  ['foo'     ]  = true,
  ['noext.t.']  = true,
  ['boring.t']  = true,
  ['.hidden' ]  = true,
  ['whitespace posix-portable name.txt'] = true
}
-- include hidden and invalid
local dir_w_all_contents = {
  ['foo'     ]  = true,
  ['noext.t.']  = true,
  ['boring.t']  = true,
  ['.hidden' ]  = true,
  ['whitespace posix-portable name.txt'] = true
}

local test_children = {}
for bn in testdir:basechildren() do
  test_children[bn] = true
end
test.seteq(test_children, dir_contents)

local test_children_hidden = {}
for bn in testdir:basechildren{show_hidden=true} do
  test_children_hidden[bn] = true
end
test.seteq(test_children_hidden, dir_w_hidden_contents)

local test_children_all = {}
for bn in testdir:basechildren{show_hidden=true, show_invalid=true} do
  test_children_all[bn] = true
end
test.seteq(test_children_all, dir_w_all_contents)


-- cleanup
os.execute('rm -r pathname_testdir')




