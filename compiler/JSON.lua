
-- This version of JSON read/write will tend to error
-- if an unsupported feature is encountered.
-- 
-- UNSUPPORTED
--    arrays
--    non-string table keys
--    unicode / getting more exotic strings right

-- HOWEVER, failures will be reported via a non-null 2nd return value
-- and an empty result rather than percolating the error up into
-- the application.  That is...
--
-- for the following calls
--    str, err_msg = JSON.stringify(val)
--    val, err_msg = JSON.parse(str)
--
-- if err_msg == nil, then no errors occurred.
-- otherwise, err_msg is the error that occurred.


local JSON = {}


-- Use the following to produce
-- deterministic order of fields being serialized
-- From the Lua Documentation
function pairs_sorted(tbl, compare)
  local arr = {}
  for k in pairs(tbl) do table.insert(arr, k) end
  table.sort(arr, compare)

  local i = 0
  local iter = function() -- iterator
    i = i + 1
    if arr[i] == nil then return nil
    else return arr[i], tbl[arr[i]] end
  end
  return iter
end


-------------------------------------------------------------------------------
--[[    STRINGIFY CODE                                                     ]]--
-------------------------------------------------------------------------------

local StringifyContext = {}
StringifyContext.__index = StringifyContext

function StringifyContext.create()
  local ctxt = setmetatable({
    visited_set = {},
    space = nil,
    space_stack = nil,
  }, StringifyContext)
  return ctxt
end

-- should only mark tables as visited
-- values may be repeated
function StringifyContext:visit( obj )
  local lookup = self.visited_set[obj]
  if lookup then
    error('detected repeat (possibly recursive?) visit to an object '..
          'during a JSON.stringify call, aborting!')
  else
    self.visited_set[obj] = true
  end
end

function StringifyContext:enable_indents( space )
  if type(space) ~= 'string' or space:find('[^ ]') then
    error('When passed space as a 2nd parameter, JSON.stringify only '..
          'accepts strings consisting of 0 or more whitespace characters')
  end
  self.space = space
  self.space_stack = { space = '', prev = nil }
end
function StringifyContext:indent()
  if self.space_stack then
    local prev = self.space_stack
    self.space_stack = {
      space = prev.space..self.space,
      prev  = prev
    }
  end
end
function StringifyContext:unindent()
  if self.space_stack and self.space_stack.prev then
    self.space_stack = self.space_stack.prev
  end
end
function StringifyContext:newline()
  if self.space_stack then
    return '\n'..self.space_stack.space
  else
    return ''
  end
end


-- namespace for mutually recursive functions
local stringify = {}

-- character substitution table for stringifying strings
local charsub = {
  ['"' ] = '\\"',
  ['\\'] = '\\\\',
  ['\b'] = '\\b',
  ['\f'] = '\\f',
  ['\n'] = '\\n',
  ['\r'] = '\\r',
  ['\t'] = '\\t',
}
-- characters to be substituted, given as a lua pattern
local charlist = '["\\\b\f\n\r\t]'

-- correctly stringify-ing and parsing strings can be dicey
-- given different encodings.  As such, please be aware that
-- this is the simplest reasonable attempt to stringify aribtrary
-- Lua strings, and not by any means a hardened routine
function stringify.string ( luastr )
  local subbed =
    luastr:gsub(charlist, function(char) return charsub[char] end)
  return '"'..subbed..'"'
end

local function is_finite ( n )
  return n > -math.huge and n < math.huge
end
function stringify.number ( luanum )
  if is_finite(luanum) then
    return tostring(luanum)
  else
    return 'null' -- if inf, -inf, or NaN
  end
end

function stringify.object ( table, ctxt )
  ctxt:visit(table)
  ctxt:indent()

  local accstr = ''
  local first = true
  for k,v in pairs_sorted(table) do
    if type(k) == 'string' then
      local keystr = stringify.string(k)
      local valstr = stringify.value(v, ctxt)

      -- comma + newline policy
      if first then
        first = false
        accstr = accstr .. ctxt:newline()
      else
        accstr = accstr .. ',' .. ctxt:newline()
      end

      -- writing the actual key-value pair out
      accstr = accstr .. keystr .. ':' .. valstr
    else
      error('The current JSON implementation ONLY allows for '..
            'serializing tables with string keys and '..
            'only string keys')
    end
  end

  ctxt:unindent()

  -- represent empty objects compactly regardles of spacing rules
  if accstr == '' then
    return '{}'
  -- non empty objects have the closing brace on a new line
  -- b/c this comes after then unindent, we get nice formatting
  else
    return '{' .. accstr .. ctxt:newline() .. '}'
  end
end

function stringify.value ( luaval, ctxt )
  local typ = type(luaval)
      if typ == 'nil' then
        return 'null'
  elseif typ == 'string' then
        return stringify.string ( luaval )
  elseif typ == 'number' then
        return stringify.number ( luaval )
  elseif typ == 'boolean' then
        if luaval then
          return 'true'
        else
          return 'false'
        end
  elseif typ == 'table' then
        return stringify.object ( luaval, ctxt )
  else
        error('unclear how we tried to JSON.stringify a lua value of type '..
              typ)
  end
end


function JSON.stringify ( luaval, space )
  local ctxt = StringifyContext.create()
  if space then
    ctxt:enable_indents(space)
  end

  local str
  local status, err = pcall(function()
    str = stringify.value( luaval, ctxt )
  end)

  if status then
    return str
  else
    return '', err
  end
end



-------------------------------------------------------------------------------
--[[    PARSE CODE                                                         ]]--
-------------------------------------------------------------------------------

local Parser = {}
Parser.__index = Parser

function Parser.create( input_str )
  local ctxt = setmetatable({
    input = input_str,
    pos   = 1,
    N     = #input_str,
  }, Parser)
  return ctxt
end

function Parser:error(msg)
  local pos_tag = self.pos
  error('JSON.parse error at position '..pos_tag..': '..msg)
end

function Parser:eof()
  return self.pos > self.N
end
-- just returns a single character
function Parser:peek()
  return self.input:sub(self.pos, self.pos)
end
-- check for string match
function Parser:matches( token )
  local tN = #token
  local snippet = self.input:sub(self.pos, self.pos + tN - 1)
  return snippet == token
end
-- advance past whatever string is expected, and error on failure
function Parser:expect( token )
  local tN = #token
  local snippet = self.input:sub(self.pos, self.pos + tN - 1)
  if snippet ~= token then
    self:error('Expected to find "'..token..'", but found "'..snippet..'"')
  else
    self.pos = self.pos + tN
  end
end
-- try to advacne past whatever string is expected,
-- but stay put and return false on failure
function Parser:nextIf( token )
  local tN = #token
  local snippet = self.input:sub(self.pos, self.pos + tN - 1)
  if snippet ~= token then
    return false
  else
    self.pos = self.pos + tN
    return true
  end
end
-- advance the read position until the specified pattern is found.
-- If the specified pattern is never found, then ERROR with msg
function Parser:nextUntil( pattern, msg )
  msg = msg or ''

  local stop = self.input:find(pattern, self.pos)

  if not stop then
    self:error(msg)
  end

  self.pos = stop
end
function Parser:nextUntilThisOrEnd( pattern )
  local stop = self.input:find(pattern, self.pos)

  if not stop then
    stop = self.N + 1
  end

  self.pos = stop
end
-- errors if the pattern cannot be matched right here
function Parser:expectPattern( pattern, msg )
  local start_pos = self.pos

  local find_pos, stop_pos = self.input:find(pattern, self.pos)
  if not find_pos or find_pos ~= start_pos then
    self:error(msg)
  end

  self.pos = stop_pos + 1
end


function Parser:whitespace()
  self:nextUntilThisOrEnd('[^%s]')
end



local evalchar_table = {
  ['\\"' ] = '"',
  ['\\\\'] = '\\',
  ['\\b' ] = '\b',
  ['\\f' ] = '\f',
  ['\\n' ] = '\n',
  ['\\r' ] = '\r',
  ['\\t' ] = '\t',
}
function Parser:evalchar( escape )
  local lookup = evalchar_table[escape]
  if not lookup then
    self:error("Found unsupported character escape sequence "..escape)
  end
  return lookup
end
-- parse out a string value
function Parser:string()
  local parser = self
  self:expect('"') -- opening
  local start_pos = self.pos

  -- advance to the closing quote
  if self:peek() == '"' then
    -- we're in the right spot already
  else
    self:nextUntil('[^\\]"', 'Expected to find closing quotatition marks')
    self.pos = self.pos + 1 -- advance to the close quotations
  end

  -- recover the string value
  local stop_pos = self.pos - 1
  local string_val = self.input:sub(start_pos, stop_pos)

  -- substitute out the special escape values
  string_val = string_val:gsub('\\.', function(escape)
    return parser:evalchar(escape)
  end)

  self:expect('"') -- closing

  return string_val
end

function Parser:number()
  local start_pos = self.pos

  -- eat an optional leading '-'
  self:nextIf('-')

  -- eat a single '0' or advance over the integral part of the number
  if not self:nextIf('0') then
    self:expectPattern('[1-9]', "Expected digit 1-9")
    self:nextUntilThisOrEnd('[^0-9]')
  end

  -- eat a decimal-point and the fractional digits if present
  if self:nextIf('.') then
    self:expectPattern('[0-9]', "Expected digit 0-9")
    self:nextUntilThisOrEnd('[^0-9]')
  end

  -- eat an exponent marker and the exponent if present
  if self:nextIf('e') or self:nextIf('E') then
    -- eat sign if present
    if not self:nextIf('+') then
      self:nextIf('-')
    end
    -- eat exponent digits
    self:expectPattern('[0-9]', "Expected digit 0-9")
    self:nextUntilThisOrEnd('[^0-9]')
  end

  local stop_pos = self.pos - 1
  if stop_pos < start_pos then
    self:error("Expected a number")
  end

  -- ok, now we can finally pull out the number string and
  -- convert it to a value
  local num_val = tonumber(self.input:sub(start_pos, stop_pos))
  return num_val
end

function Parser:object()
  local obj_val = {}
  self:expect('{')
  self:whitespace()

  local function keyval(parser)
    local key = parser:string()
    parser:whitespace()
    parser:expect(':')
    parser:whitespace()
    local val = parser:value()
    obj_val[key] = val
  end

  -- read first entry
  if self:peek() ~= '}' then
    keyval(self)
    self:whitespace()
  end

  -- and the rest
  while self:peek() ~= '}' do
    self:expect(',')
    self:whitespace()
    keyval(self)
    self:whitespace()
  end

  self:whitespace()
  self:expect('}')
  return obj_val
end

function Parser:value()
  if self:eof() then
    self:error('Expected JSON value, but reached end of file')
  end

  local char = self:peek()

      if char == '"' then
        return self:string()
  elseif char:find('[-0-9]') then
        return self:number()
  elseif char == '{' then
        return self:object()
  elseif char == '[' then
    self:error('Current JSON.parse does not support arrays')
  elseif char == 't' then
        self:expect('true')
        return true
  elseif char == 'f' then
        self:expect('false')
        return false
  elseif char == 'n' then
        self:expect('null')
        return nil
  else
    self:error('Expected JSON value, but found character "'..char..'"')
  end
end

function Parser:json()
  self:whitespace()
  local value = self:value()
  self:whitespace()

  if not self:eof() then
    self:error("expected to see one value in JSON string, "..
                 "but there was more")
  end

  return value
end

function JSON.parse ( json_str )
  local parser = Parser.create( json_str )

  local value
  local status, err = pcall(function()
    value = parser:json()
  end)

  if status then
    return value
  else
    return nil, err
  end
end





--local garble = {
--  x = {
--    y = "blah"
--  },
--  num = 123,
--}
--
--
--print(JSON.stringify(garble))
--print(JSON.stringify( garble, '  '))
--print(JSON.stringify( garble, '' ))
--
--local str1 = '{"num":123,"x":{"y":"blah"}}'
--local str2 = [[{
--  "num":123.4E-5,
--  "x":{
--    "y":"blah"
--  }
--}]]
--local str3 = [[{
--"num":123,
--"x":{
--"y":"blah"
--}
--}]]
--
--print(JSON.stringify(JSON.parse(str1)))
--print(JSON.stringify(JSON.parse(str2)))
--print(JSON.stringify(JSON.parse(str3)))
--
--local gobj = JSON.parse(JSON.stringify(garble))
--local function compareobj(obj1, obj2)
--  if type(obj1) == 'table' then
--    if type(obj2) ~= 'table' then print('t fail') return false end
--    -- check every key in obj2 is in obj1
--    for k, _ in pairs(obj2) do
--      if obj1[k] == nil then
--        return false
--      end
--    end
--    -- now check every key in obj1 is in obj2 and matches values
--    for k, v1 in pairs(obj1) do
--      local v2 = obj2[k]
--      if not compareobj(v1, v2) then
--        return false
--      end
--    end
--    -- successfully compared tables
--    return true
--  else
--    return obj1 == obj2
--  end
--end
--print(compareobj(gobj, garble))
--local function printobj(obj, indent)
--  indent = indent or ''
--  if type(obj) == 'table' then
--    for k,v in pairs(obj) do
--      print(indent..tostring(k))
--      printobj(v, indent..'  ')
--    end
--  else
--    print(indent..tostring(obj))
--  end
--end
--printobj(gobj)
--printobj(garble)



return JSON



