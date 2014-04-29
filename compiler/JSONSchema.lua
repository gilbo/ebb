

local JSONSchema = {}
JSONSchema.__index = JSONSchema
package.loaded["compiler.JSONSchema"] = JSONSchema

local JSON = terralib.require 'compiler.JSON'


-------------------------------------------------------------------------------
--[[    Schema Nodes                                                       ]]--
-------------------------------------------------------------------------------
local function newProto(proto)
  local tbl = {}
  if proto then setmetatable(tbl, proto) end
  tbl.__index = tbl
  return tbl
end
JSONSchema.Schema   = newProto()
JSONSchema.Object   = newProto(JSONSchema.Schema)
JSONSchema.Num      = newProto(JSONSchema.Schema)
JSONSchema.Literal  = newProto(JSONSchema.Schema)
JSONSchema.String   = newProto(JSONSchema.Schema)
JSONSchema.OrNode   = newProto(JSONSchema.Schema)

function JSONSchema.is_schema(val)
  local p = val
  while p ~= nil do
    if p == JSONSchema.Schema then return true end
    p = getmetatable(p)
  end
  return false
end

-------------------------------------------------------------------------------
--[[    Schema Building                                                    ]]--
-------------------------------------------------------------------------------

function JSONSchema.OR(option_list)
  local n_opt = #option_list

  -- list of alternatives to try
  local opts = {}
  for i=1,n_opt do
    opts[i] = JSONSchema.New(option_list[i])
  end

  local or_node = setmetatable({
    options = opts
  }, JSONSchema.OrNode)
  return or_node
end

function JSONSchema.New(pattern)
  if JSONSchema.is_schema(pattern) then return pattern end

  if type(pattern) == 'number' or type(pattern) == 'string' then
    local literal = setmetatable({
      value = pattern
    }, JSONSchema.Literal)
    return literal
  end

  if type(pattern) == 'table' then
    local wildcard = nil
    local keys     = {}
    local optional = {}

    for k,v in pairs(pattern) do
      if k == '*' then
        wildcard = JSONSchema.New(v)
      elseif k == '?' then
        if type(v) ~= 'table' then
          error('? must map to a table of optional fields', 2)
        end
        -- read out all the options
        for optk,optv in pairs(v) do
          optional[optk] = JSONSchema.New(optv)
        end
      else
        keys[k] = JSONSchema.New(v)
      end
    end

    local obj = setmetatable({
      keys = keys, -- mapping of keys to sub-schema
      optional = optional, -- keys that are optional...
    }, JSONSchema.Object)
    -- optional default case schema for further field names
    if wildcard then obj.wildcard = wildcard end

    return obj
  end


  error('could not figure out how to convert JSONSchema.New argument '..
        'to a schema: '..tostring(pattern), 2)
end

-------------------------------------------------------------------------------
--[[    Error Reporting                                                    ]]--
-------------------------------------------------------------------------------

local ErrorReport = {}
ErrorReport.__index = ErrorReport

function ErrorReport.New()
  local report = setmetatable({
    messages = {},
    location = nil, -- { up = ..., loc = "..." }
    suppressed = false
  }, ErrorReport)
  return report
end

function ErrorReport:suppress()
  self.suppressed = true
end
function ErrorReport:unsuppress()
  self.suppressed = false
end

function ErrorReport:pushLocation(loc)
  local up = self.location
  self.location = { up = up, loc = loc }
end
function ErrorReport:popLocation()
  if self.location then
    self.location = self.location.up
  end
end

function ErrorReport:locstring()
  local location = self.location
  local str = ''
  while location do
    str = '.' .. location.loc .. str
    location = location.up
  end
  return '<root>' .. str
end

function ErrorReport:error(msg)
  if not self.suppressed then
    msg = self:locstring() .. ': ' .. msg
    table.insert(self.messages, msg)
  end
end


-------------------------------------------------------------------------------
--[[    Schema Matching                                                    ]]--
-------------------------------------------------------------------------------

function JSONSchema.match(schema, instance, errors)
  if not JSONSchema.is_schema(schema) then
    error('Must pass a schema as the first argument to JSONSchema.match()', 2)
  end

  local  report = ErrorReport.New()
  local  result = schema:match(instance, report)
  if type(errors) == 'table' then
    for _,msg in ipairs(report.messages) do table.insert(errors, msg) end
  end

  return result
end

function JSONSchema.Num:match(num, report)
  local  result = type(num) == 'number'
  if not result then report:error('expected number') end
  return result
end
function JSONSchema.String:match(str, report)
  local  result = type(str) == 'string'
  if not result then report:error('expected string') end
  return result
end
function JSONSchema.Literal:match(literal, report)
  local  result = self.value == literal
  if not result then report:error("expected value "..tostring(self.value)) end
  return result
end
function JSONSchema.Object:match(tbl, report)
  -- type it
  if type(tbl) ~= 'table' then
    report:error("expected object")
    return false
  end

  local success = true

  -- must find all the keys and match them
  for k,v in pairs(self.keys) do
    local lookup = tbl[k]
    if lookup == nil then
      report:error("expected to find key '"..k.."'")
      success = false
    else
      report:pushLocation(k)
      if not v:match(lookup, report) then
        success = false
      end
      report:popLocation()
    end
  end

  -- use the wildcard to handle all the rest of the fields present
  for k,v in pairs(tbl) do
    if not self.keys[k] then
      -- found non-string key
      if type(k) ~= 'string' then
        report:error("expected key to be a string")
        success = false
      -- found optional key
      elseif self.optional[k] then
        report:pushLocation(k)
        if not self.optional[k]:match(v, report) then
          success = false
        end
        report:popLocation()
      -- found default key handler
      elseif self.wildcard then
        report:pushLocation(k)
        if not self.wildcard:match(v, report) then
          success = false
        end
        report:popLocation()
      else
        report:error("did not expect to find key '"..k.."'")
        success = false
      end
    end
  end

  -- in the absence of errors we're good
  return success
end
function JSONSchema.OrNode:match(val, report)
  local success = false

  report:suppress()
  for _,subschema in ipairs(self.options) do
    if subschema:match(val, report) then 
      success = true 
      break
    end
  end
  report:unsuppress()

  if not success then
    report:error('none of the options worked')
  end

  return success
end




return JSONSchema