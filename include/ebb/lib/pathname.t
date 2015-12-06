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


-- Pathname is a collection of handy helper routines for mangling
-- strings that are meant to represent filesystem paths.

-- It is roughly inspired by Ruby's Pathname module, which
-- is relatively user friendly.

-- One works with "pathname" objects instead of raw strings
-- with easy conversion from/to raw strings

-- NOTE NOTE: Pathnames are assumed to be immutable
-- If any routine in this module tries to mutate a pathname
-- rather than making a copy and mutating the copy, then
-- that's definitely a BIG ERROR.


local PN = {}
package.loaded["ebb.lib.pathname"] = PN

local ffi = require("ffi")
local C   = require "ebb.src.c"

-- All this garble is needed to expose
-- POSIX routines in a workable form
local sys = terralib.includecstring([[
// POSIX related headers
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

// error codes if we get more sophisticated here
//#include <errno.h>

// get the current directory with this call
// (caller responsible for freeing returned memory)
// char *getcwd(NULL, 0);

// use this call to find out about a file
// all uses of this function are in the C code below
// int stat(const char *path, struct stat *buf);

// if the stat call failed (non-zero return code),
// then we're going to somewhat incorrectly interpret this
// as the file in question not existing
// The real problem could be access control or otherwise
int path_exists(const char *path) {
  struct stat buf;
  int err = stat(path, &buf);
  // report errors as boolean return value
  if(err == 0) return 1; // true
  else         return 0; // false
}

// by inspecting buf.st_mode, we can find out...
int path_is_file(const char *path) {
  struct stat buf;
  int err = stat(path, &buf);
  if(err != 0)
    return 0; // false
  return S_ISREG(buf.st_mode); // is this a regular file?
}
int path_is_dir(const char *path) {
  struct stat buf;
  int err = stat(path, &buf);
  if(err != 0)
    return 0; // false
  return S_ISDIR(buf.st_mode); // is this a directory?
}

// we can make a directory like so...
int mkdir_wrapper(const char *path) {
  return mkdir(path, S_IRWXG | S_IRWXU | S_IRWXO);
}

// we can get the contents of a directory by getting a handle:
//    DIR* opendir(void)
//    int closedir(DIR*)
// We can supply this handle to the following helper function
// to sequentially retreive filenames.  Make sure to save a copy
// of the filename after each call, before the next.
// When NULL is returned, there are no more children to read.
char* readdir_str(DIR *dirp) {
  struct dirent *filedata = readdir(dirp);
  if(filedata)  return filedata->d_name;
  else          return NULL;
}

]])


------------------------------------------------------------------------------
-- Reference Path Strings: Ebb Root & Working Directory                     --
------------------------------------------------------------------------------

-- compute the current working directory...
local WORKING_DIRECTORY_CACHED
(function()
  local strptr = sys.getcwd(nil,0)
  if strptr ~= nil then
    WORKING_DIRECTORY_CACHED = ffi.string(strptr)
    C.free(strptr)
  end
end)()
if not WORKING_DIRECTORY_CACHED or #WORKING_DIRECTORY_CACHED == 0 then
  error('Failed to determine working directory.  Aborting system')
end

-- Compute the Ebb root directory
-- Super Hacky!
local EBB_RELEASE_DIR_CACHED
(function()
  local info = debug.getinfo(1, "S")
  local src  = info.source
  -- strip leading '@' and trailing '/compiler/pathname.t'
  local strip = ('ebb/lib/pathname.t'):len()
  EBB_RELEASE_DIR_CACHED = src:sub(2,-strip-1)
  if EBB_RELEASE_DIR_CACHED:sub(1,1) == '.' then
    EBB_RELEASE_DIR_CACHED = WORKING_DIRECTORY_CACHED..
                            (EBB_RELEASE_DIR_CACHED:sub(2))
  end
end)()

-- functions providing absolute paths as strings
PN.pwd_str = function() return WORKING_DIRECTORY_CACHED end
PN.ebb_root_str = function() return EBB_RELEASE_DIR_CACHED end


------------------------------------------------------------------------------
-- Pathname String Input Mangling & Validation                              --
------------------------------------------------------------------------------

local function split_pathstr(str)
  local tokens = {}
  -- if the first character is a slash, then grab that
  if str:sub(1,1) == '/' then
    table.insert(tokens, '/')
  end
  -- get the rest of the tokens ignoring repeated slashes
  for tkn in str:gmatch('[^/]+') do
    table.insert(tokens, tkn)
  end
  return tokens
end

-- We attempt to ensure maximum interoperability of filenames
local POSIX_name_rules_text = [[
Valid filename / filepath strings:
  This system attempts to prevent problems with filenames and filepaths
  by being conservative about what it considers a valid file string.
  The rules for filenames are...
    1. Alphanumeric characters are allowed (i.e. a-z, A-Z and 0-9)
    2. Underscores (i.e. _ ), dots (i.e. . ) and hyphens (i.e. - ) are allowed
    3. Hyphens are not allowed as the first character of a filename
    4. Spaces are allowed, but no leading or trailing whitespace is allowed
  File Paths are sequences of filenames, separated by
    forward slashes (i.e. / ), repetition allowed.
    If the first character is a slash, then the path starts
    at the root directory.

  We will not check whether or not a name is exactly 'null', but
    YOU SHOULD NOT DO THAT!
  (note: these are roughly the POSIX portable filenames plus spaces)
]]
local function POSIX_valid_filename(str)
  local test = not not str:match('^[%w%._][%w%.%s_-]*$')
  return test and not str:match('%s$') -- and does not end with a space
end
-- ALLOWS for empty paths, should check another way!
local function POSIX_valid_pathname(str)
  local tokens = split_pathstr(str)
  if #tokens < 1 then return true end

  -- check first token, which we allow to designate the root optionally
  if not POSIX_valid_filename(tokens[1]) and tokens[1] ~= '/' then
    return false
  end

  -- check the remaining tokens
  for i = 2,#tokens do
    if not POSIX_valid_filename(tokens[i]) then
      return false
    end
  end

  return true
end


------------------------------------------------------------------------------
-- Pathname Type Declaration and Means of Construction                      --
------------------------------------------------------------------------------

local Pathname    = {}
Pathname.__index  = Pathname
PN.Pathname       = Pathname

-- type checking
function PN.is_pathname (obj)
  return getmetatable(obj) == Pathname
end


-- helper to convert strings to pathnames
local function path_from_str(path_str, err_lvl)
  if not POSIX_valid_pathname(path_str) then
    error('Bad Pathname:\n'..'"'..path_str..'"\n'..
          POSIX_name_rules_text, err_lvl)
  end

  local tokens = split_pathstr(path_str)
  local has_root_token = false
  if tokens[1] == '/' then
    has_root_token = true
    table.remove(tokens, 1)
  end

  local pathname = setmetatable({
    has_root_token  = has_root_token,
    tokens          = tokens
  }, Pathname)

  return pathname
end
-- helper to make sure that all Pathname functions are non-destructive
local function clone(path)
  local copy = setmetatable({
    has_root_token  = path.has_root_token,
    tokens          = {}
  }, Pathname)
  for i,v in ipairs(path.tokens) do copy.tokens[i] = v end
  return copy
end
-- helper to allow for strings to be passed as args into pathname functions
local function path_or_str(obj)
  if type(obj) == 'string' then
    obj = path_from_str(obj, 4)
  elseif not PN.is_pathname(obj) then
    error('Expected pathname argument, but got neither a '..
          'pathname, nor a string', 3)
  end

  return obj
end


-- General exposed interface for constructing new pathnames from strings
function Pathname.new(path_str)
  if PN.is_pathname(path_str) then
    return path_str
  end
  if type(path_str) ~= 'string' then
    error('Pathname.new() expects a string argument', 2)
  end

  return path_from_str(path_str, 2)
end

------------------------------------------------------------------------------
-- Pathname Conversion to String                                            --
------------------------------------------------------------------------------

function Pathname:tostring()
  local str = ''
  if self:is_absolute() then str = '/' end
  if #self.tokens == 0 then return '.' end

  if #self.tokens > 0 then str = str..self.tokens[1] end
  for i = 2,#self.tokens do
    str = str..'/'..self.tokens[i]
  end

  return str
end
Pathname.__tostring = Pathname.tostring


------------------------------------------------------------------------------
-- Pathname Constants                                                       --
------------------------------------------------------------------------------

-- throw a comprehensible error if the ebb_root contains
-- non POSIX-portable filenames along the way.  Right now,
-- let's just error on this case.  We may be forced to support
-- a wider class of valid pathnames later to prevent users complaining.
if not POSIX_valid_pathname(PN.ebb_root_str()) then
  error(POSIX_name_rules_text.."\n"..
        "The current installation path for Ebb is not POSIX portable.\n"..
        "The current installation path is... \n"..
        PN.ebb_root_str().."\n"..
        "\n"..
        "Please contact the Ebb developers if this is a serious problem.")
end

Pathname.root         = Pathname.new('/')
Pathname.pwd          = Pathname.new(PN.pwd_str())
Pathname.ebb_root     = Pathname.new(PN.ebb_root_str())

function PN.root        ()  return Pathname.root        end
function PN.pwd         ()  return Pathname.pwd         end
function PN.ebb_root    ()  return Pathname.ebb_root    end
PN.getwd = PN.pwd

function Pathname.scriptdir()
  -- get the source path of the calling function
  local info = debug.getinfo(2, "S")
  local src  = info.source
  -- strip leading '@'
  return Pathname.new(src:sub(2,-1)):dirpath()
end
PN.scriptdir = Pathname.scriptdir


------------------------------------------------------------------------------
-- Pathname Tests                                                           --
------------------------------------------------------------------------------

-- test whether the pathname is absolute or relative
function Pathname:is_absolute()
  return self.has_root_token
end
function Pathname:is_relative()
  return not self.has_root_token
end

-- test whether this pathname refers to the root directory
-- WARNING: Will not detect '/usr/..' etc.
function Pathname:is_root()
  return self.has_root_token and #self.tokens == 0
end


------------------------------------------------------------------------------
-- Pathname Decomposition                                                   --
------------------------------------------------------------------------------

-- get the basename (in UNIX sense)
-- i.e. the last file in the pathname sequence
-- The basename will be returned as a string
-- If this is a root pathname, then the empty string will be returned
function Pathname:basename()
  if self:is_root() then return '' end
  return self.tokens[#self.tokens]
end

-- get the dirpath (everything but the basename)
-- The dirpath will be returned as a pathname
function Pathname:dirpath()
  if self:is_root() then return self end
  local pn = clone(self)
  table.remove(pn.tokens) -- removes last element
  return pn
end

-- split splits the path into a dirpath followed by the basename
function Pathname:split()
  return self:dirpath(), self:basename()
end

-- Returns everything after the last period in the basename
-- UNLESS:
--    the basename begins or ends with a period
--    there is no period in the basename
-- In these failure cases, the empty string is returned
function Pathname:extname()
  local base = self:basename()
  if base:sub(1,1) == '.' then return '' end

  local ext  = base:match('%.([^%.]+)$')
  if ext then return ext
  else        return '' end
end

-- dirname will get the dirpath as a string
function Pathname:dirname()
  return self:dirpath():tostring()
end

-- absolute dir path/name
function Pathname:absdirpath()
  return self:abspath():dirpath()
end
function Pathname:absdirname()
  return self:abspath():dirname()
end

------------------------------------------------------------------------------
-- Pathname Composition                                                     --
------------------------------------------------------------------------------

-- concatenate a pathname to this one
-- if rhs is a string, convert it to a pathname
-- Will fail if rhs is an absolute path
function Pathname.concat(lhs, rhs)
  lhs = path_or_str(lhs)
  rhs = path_or_str(rhs)
  if rhs:is_absolute() then
    error("cannot concatenate an absolute path on the right", 2)
  end
  local pn = clone(lhs)
  for i,v in ipairs(rhs.tokens) do table.insert(pn.tokens, v) end
  return pn
end
Pathname.__concat = Pathname.concat

-- return a clean path to the parent directory
function Pathname:parent()
  local pn = clone(self)
  table.insert(pn.tokens, '..')
  return pn:cleanpath()
end

-- absolute path
function Pathname:abspath()
  if self:is_absolute() then
    return self
  else
    return Pathname.pwd:concat(self)
  end
end

-- collapse out .. and . segments from the path
-- May exhibit strange behavior around symlinks
function Pathname:cleanpath()
  local pn = clone(self)
  local i  = 1
  while i <= #pn.tokens do
    local tkn = pn.tokens[i]
    if tkn == '.' then -- remove dots
      table.remove(pn.tokens, i)
    elseif tkn == '..' then -- collapse out double dots if possible
      if i > 1 and pn.tokens[i-1] ~= '..' then
        table.remove(pn.tokens, i)
        table.remove(pn.tokens, i-1)
        i = i - 1
      else
        i = i + 1
      end
    else -- not a special token, advance
      i = i + 1
    end
  end
  return pn
end


------------------------------------------------------------------------------
-- Functions that rely on system calls                                      --
------------------------------------------------------------------------------


-- helper function that retreives a list of the directory's contents
local function child_list(dirstr)
  local dir = sys.opendir(dirstr)
  if dir == nil then return {} end

  local children = {}

  -- step through the children and pack them into the list
  local childstr = sys.readdir_str(dir)
  while childstr ~= nil do
    table.insert(children, ffi.string(childstr))
    childstr = sys.readdir_str(dir)
  end
  sys.closedir(dir)

  return children
end

-- params accepts the following parameters:
--    show_hidden = true  (will include files begining with a dot)
--    show_invalid = true  (will include files with non POSIX-valid names)
local function child_iter(dirstr, params)
  local children = child_list(dirstr)
  local i = 1
  return function()
    repeat
      local child = children[i]
      i = i + 1

      -- filter
      if child == '.' or child == '..' then
        child = nil
      end
      if child and not params.show_hidden then -- hidden filtering
        if child:sub(1,1) == '.' then
          child = nil
        end
      end
      if child and not params.show_invalid then -- invalid filtering
        if not POSIX_valid_filename(child) then
          child = nil
        end
      end

      -- report any unfiltered children
      if child then return child end
    until i > #children -- force exit eventually
  end
end

-- iterator function for iterating over childrens' pathnames
function Pathname:children(params)
  params = params or {}
  -- do not allow show invalid
  params.show_invalid = nil

  local path = self
  local iter = child_iter(self:tostring(), params)

  return function() -- iterator function wrapper
    local basechild = iter()
    if basechild then
      return path.concat(basechild)
    end
  end
end
-- iterator function for iterating over childrens' basenames
function Pathname:basechildren(params)
  params = params or {}
  return child_iter(self:tostring(), params)
end



function Pathname:is_directory()
  return 0 ~= sys.path_is_dir(self:tostring())
end
function Pathname:is_file()
  return 0 ~= sys.path_is_file(self:tostring())
end
-- tests whether the described file/directory exists
function Pathname:exists()
  return 0 ~= sys.path_exists(self:tostring())
end
-- test whether the dirpath exists
function Pathname:direxists()
  return self:dirpath():exists()
end

-- attempts to create the specified directory on disk
-- returns true on successful creation, false on failure
-- (this could be b/c the directory already exists)
-- you can test whether it's already there with another call!
function Pathname:mkdir()
  local err = sys.mkdir_wrapper(self:tostring())
  return err == 0
end
-- attempts to create the specified directory,
-- but will also happily create any necessary
-- intermediary directories along the path...
function Pathname:mkpath()
  local exists = self:exists()
  if #self.tokens > 0 and not self:exists() then
    local success = self:parent():mkpath()
    if not success then return success end
    return self:mkdir()
  end
  return true -- success
end






