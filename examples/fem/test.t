import 'compiler.liszt'

function split_on_spaces(str)
  local a = {}
  while str:len() > 0 do
    -- trim any whitespace at the front
    str = str:sub(str:match("^%s*"):len()+1) -- 1 indexing
    -- read off the next token
    local token = str:match("[^%s]+")
    if token == nil then break end  -- break if there isn't a token
    -- save the token
    table.insert(a,token)
    -- advance past token
    str = str:sub(token:len()+1) -- 1 indexing
  end
  return a
end


function dump_list(t)
  print("size is "..tostring(#t))
  for i=1,#t do
    print(t[i])
  end
end

--dump_list(split_on_spaces("asdf"))

local Tetmesh = L.require 'examples.fem.tetmesh'
local VEGFileIO = L.require 'examples.fem.vegfileio'

local turtle = VEGFileIO.LoadTetmesh
  'examples/fem/turtle-volumetric-homogeneous.veg'



