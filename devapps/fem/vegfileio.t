import "ebb"
local L = require "ebblib"

local VEGFileIO = {}
VEGFileIO.__index = VEGFileIO
package.loaded["devapps.fem.vegfileio"] = VEGFileIO


local Tetmesh = require 'devapps.fem.tetmesh'


------------------------------------------------------------------------------

-- VEG files have the following format
--
--  # this is a comment
--  
--  *KIND_OF_DATA_SECTION (i.e. VERTICES, ELEMENTS)
--  ...
--
--  *VERTICES
--  n_vertices n_coords_per_line n_???=0 n_???=0
--  index# x_position y_position ...
--  ... (We assume indices are contiguous starting from 1)
--  
--  *ELEMENTS
--  KIND_OF_ELEMENT (Only TET supported right now)
--  n_elements n_vertices_per_element n_???=0
--  index# vert_id_1 vert_id_2 ...
--  ... (We assume indices are contiguous starting from 1)
--

------------------------------------------------------------------------------

local function split_on_spaces(str)
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

local function toint(str)
  local num = tonumber(str)
  if num ~= nil and math.floor(num) == num then return num
                                           else return nil end
end
local function arr_toint(xs)
  local rs = {}
  for i=1,#xs do
    rs[i] = toint(xs[i])
  end
  return rs
end
local function has_nil(xs)
  for i=1,#xs do
    if xs[i] == nil then return true end
  end
  return false
end

------------------------------------------------------------------------------

function VEGFileIO.LoadTetmesh(path)
  -- make sure path is converted to a string before use
  path = tostring(path)

  -- In Lua, we can open files just like in C
  local infile = io.open(path, "r")
  if not infile then
    error('failed to open VEG file '..path)
  end

  -- temporaries and results of reading
  local section_mode = nil
  -- VERTICES
  local n_vertices = nil -- metadata
  local vertex_counter = 0 -- observed count
  local vertices = {}
  -- ELEMENTS (TET)
  local elem_type = nil -- will have to be TET
  local n_elements = nil -- metadata
  local elem_counter = 0 -- observed count
  local elements = {}
  -- MATERIAL (ENU)
  local material = nil -- will have to be ENU
  local density = nil
  local E = nil
  local Nu = nil

  -- Read Loop
  local line_no = 0
  local line = infile:read()
  local function report_error(msg)
    error('Found error while reading VEG file '..
          path..'@'..tostring(line_no)..':\n'..msg,2)
  end
  while line ~= nil do

    -- TRIM WHITE SPACE AT FRONT
    line = line:sub(line:match("^%s*"):len() + 1)

    local first_char = line:sub(0,1)

    -- BLANK LINES (length 0)
    if first_char:len() == 0 then
      -- skip

    -- COMMENT LINES
    elseif first_char == '#' then
      --skip

    -- NEW SECTION DIRECTIVES
    elseif first_char == '*' then
      local title = line:sub(2) -- 1-based indexing
      local args = split_on_spaces(title)

      section_mode = args[1]
      if args[1] == 'VERTICES' then
        if #args > 1 then report_error('Expected 0 args to VERTICES') end
      elseif args[1] == 'ELEMENTS' then
        if #args > 1 then report_error('Expected 0 args to ELEMENTS') end
      elseif args[1] == 'MATERIAL' then
        -- Do nothing for now, but allow
      elseif args[1] == 'REGION' then
        -- Do nothing for now, but allow
      else
        report_error('Unrecognized section name: '..section_mode)
      end

    -- Otherwise, behavior is dependent on which mode we're in
    elseif section_mode == 'VERTICES' then
      local tokens = split_on_spaces(line)
      -- Metadata line
      if n_vertices == nil then
        local ints = arr_toint(tokens)
        if has_nil(ints) or #ints ~= 4 then
          report_error('Was expecting 4 integers')
        end
        n_vertices = ints[1]
        if ints[2] ~= 3 then
          report_error('Was expecting 3d points')
        end
        if ints[3] ~= 0 or ints[4] ~= 0 then
          report_error('Was expecting the last two nums to be 0')
        end
      -- Data line
      else
        local index = toint(tokens[1])
        local x = tonumber(tokens[2])
        local y = tonumber(tokens[3])
        local z = tonumber(tokens[4])
        if #tokens ~= 4 or not index or not x or not y or not z then
          report_error('Was expecting 4 numbers per line (first is int)')
        end
        if index ~= vertex_counter+1 then
          report_error('Expected contiguous vertex indices starting at 1')
        end
        -- save the vertex data
        vertex_counter = vertex_counter+1 -- Lua indexing starts at 1
        vertices[vertex_counter] = {x,y,z}
      end

    elseif section_mode == 'ELEMENTS' then
      local tokens = split_on_spaces(line)
      -- Type line
      if elem_type == nil then
        elem_type = line
        if elem_type ~= 'TET' then
          report_error('Only supporting TET elements')
        end
      -- Metadata line
      elseif n_elements == nil then
        local ints = arr_toint(tokens)
        if has_nil(ints) or #ints ~= 3 then
          report_error('Was expecting 3 integers')
        end
        n_elements = ints[1]
        if ints[2] ~= 4 then
          report_error('Was expecting 4 vertices per element')
        end
        if ints[3] ~= 0 then
          report_error('Was expecting the last num to be 0')
        end
      -- Data line
      else
        local ints = arr_toint(tokens)
        if has_nil(ints) or #ints ~= 5 then
          report_error('Was expecting 5 ints per line (first is index)')
        end
        if ints[1] ~= elem_counter+1 then
          report_error('Expected contiguous vertex indices starting at 1')
        end
        -- save the index data
        elem_counter = elem_counter+1 -- Lua indexing starts at 1
        elements[elem_counter] = {ints[2]-1,ints[3]-1,ints[4]-1,ints[5]-1}
      end

    elseif section_mode == 'MATERIAL' then
      local tokens = split_on_spaces(line)
      if has_nil(tokens) or #tokens ~= 4 then
        report_error('Was expecting 4 tokens for material specification')
      end
      material = tokens[1]:sub(1, tokens[1]:len()-1)
      if material ~= 'ENU' then
        report_error('Only supporting ENU material')
      end
      density = tonumber(tokens[2]:sub(1, tokens[2]:len()-1))
      E = tonumber(tokens[3]:sub(1, tokens[3]:len()-1))
      Nu = tonumber(tokens[4])

    elseif section_mode == 'REGION' then
      -- Do Nothing for now...

    else
      report_error('Found non-empty, non-comment line outside of a *SECTION')
    end

    line = infile:read()
    line_no = line_no+1
  end

  -- don't forget to close the file when done
  infile:close()

  -- check completeness
  if #vertices ~= n_vertices then
    report_error('File claimed that there are '.. tostring(n_vertices) ..' vertices, '..
                 'but we found '.. tostring(#vertices) ..' vertices')
  end
  if #elements ~= n_elements then
    report_error('File claimed that there are '.. tostring(n_elements) ..' elements, '..
                 'but we found '.. tostring(#elements) ..' elements')
  end

  local mesh = Tetmesh.LoadFromLists(vertices, elements)
  mesh.density = density
  mesh.E = E
  mesh.Nu = Nu
  return mesh
end




