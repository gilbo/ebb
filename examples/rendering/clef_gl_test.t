import "compiler.liszt"

--[[

    This file builds on the core GLFW test, by refactoring out useful
    abstractions for OpenGL.

]]--

local ffi = require 'ffi'
local gl  = require 'gl.gl'
local VO = require 'gl.vo'
local dld = require 'compiler.dld'
local mat4f = require 'gl.mat4f'

local C = require 'compiler.c'
local vdb = L.require 'lib.vdb'

local glshader = require 'gl.shader'

-- patch in gclef to the relations
require 'gl.gclef'

print(mat4f.id() * mat4f.id())

-----------------------------------------------------------------------------
--[[                             GLFW SETUP                              ]]--
-----------------------------------------------------------------------------


-- Route GLFW errors to the console
gl.glfwSetErrorCallback(function(err_code, err_str)
    io.stderr:write('GLFW ERROR; code: '..tostring(err_code)..'\n')
    io.stderr:write(ffi.string(err_str) .. '\n')
end)

-- init glfw
if not gl.glfwInit() then
    error('failed to initialize GLFW')
end

-- request a GL Core (forward compatible) Context
-- Keep it simple to demonstrate ONLY GLFW here
gl.requestCoreContext()



-- create the window
local window = gl.glfwCreateWindow(640, 480, "Hello World", nil, nil);
if not window then
    gl.glfwTerminate()
    error('Failed to create GLFW Window')
end
gl.glfwMakeContextCurrent(window)

-- example input callback
-- Close on ESC
gl.glfwSetKeyCallback(window,
function(window, key, scancode, action, mods)
    if key == gl.GLFW_KEY_ESCAPE_val() and
       action == gl.GLFW_PRESS_val()
    then
        gl.glfwSetWindowShouldClose(window, true)
    end
end)



-- NOTE THAT THE MAIN ISSUE IS BINDING
-- THE CHOICE HERE IS WHICH "SLOT" or number
-- each field/attribute gets bound to.
-- The names should be determined rather trivially.
-- May want name aliasing (that seems maybe a tad much??? maybe not)
-- Definitely want to be able to subset which of all the fields we export



-----------------------------------------------------------------------------
--[[                Setup Liszt Relations & init GClef                   ]]--
-----------------------------------------------------------------------------

local triangles = L.NewRelation(1, 'triangles')
local vertices  = L.NewRelation(3, 'vertices')

triangles:NewField('v0', vertices):Load(function(i) return 3*i + 0 end)
triangles:NewField('v1', vertices):Load(function(i) return 3*i + 1 end)
triangles:NewField('v2', vertices):Load(function(i) return 3*i + 2 end)

local vpos = terralib.global(float[1 * 3 * 4])
vpos:set({
 -0.6, -0.4, 0, 1,
  0.6, -0.4, 0, 1,
    0,  0.6, 0, 1
})
vertices:NewField('pos', L.vector(L.float, 4)):Load(function(i)
    local p = vpos:get()
    return {p[4*i+0], p[4*i+1], p[4*i+2], p[4*i+3]}
end)


local vcolor = terralib.global(float[1 * 3 * 3])
vcolor:set({
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
})

vertices:NewField('color', L.vector(L.float, 3)):Load(function(i)
    local c = vcolor:get()
    return {c[3*i+0], c[3*i+1], c[3*i+2]}
end)

local position_attr_id = 0
local color_attr_id = 1

--for k,v in pairs(getmetatable(triangles)) do print(k,v) end
triangles:CreateGClef {
    triangle = { 'v0', 'v1', 'v2' },
    attr_ids = {
        pos   = position_attr_id,
        color = color_attr_id,
    }
}

-----------------------------------------------------------------------------
--[[                             INIT SHADER                             ]]--
-----------------------------------------------------------------------------

local shader = glshader.new()

shader:load_vert_str([[#version 330

//layout (std140) uniform Matrices {
//    mat4 pvm;
//} ;
uniform mat4 pvm;

in vec4 position;
in vec4 vcolor;

out vec4 color;

void main()
{
    color = vec4(vcolor);
    gl_Position = pvm * position ;
} 
]])

shader:load_frag_str([[#version 330

in vec4 color;

out vec4 outputF;

void main()
{
    outputF = color;
} 
]])

shader:compile()

gl.glBindFragDataLocation(shader:id(), 0, 'outputF')

gl.glBindAttribLocation(shader:id(), position_attr_id, 'position')
gl.glBindAttribLocation(shader:id(), color_attr_id, 'vcolor')

shader:link()


print('testing shader validity...')
if not shader:validate(triangles:GetGClefVO()) then
    print('WARNING WARNING: Shader is NOT VALID')
else
    print('shader valid')
end

local logs = shader:getLogs({ null_val = 'no info' })

print(
[[/--------------------\
|vert shader info log|
\--------------------/
]]..logs.vert)
print(
[[/--------------------\
|frag shader info log|
\--------------------/
]]..logs.frag)
print(
[[/-----------------------\
|shader program info log|
\-----------------------/
]]..logs.prog)

print('')

--[[
1280 GL_INVALID_ENUM      
1281 GL_INVALID_VALUE     
1282 GL_INVALID_OPERATION 
1283 GL_STACK_OVERFLOW    
1284 GL_STACK_UNDERFLOW   
1285 GL_OUT_OF_MEMORY
]]--
print('gl error code (see source; 0 is all ok): ', gl.glGetError())


-----------------------------------------------------------------------------
--[[                             INIT OPENGL                             ]]--
-----------------------------------------------------------------------------


-- INIT OPENGL
gl.glEnable(gl.DEPTH_TEST)
gl.glEnable(gl.CULL_FACE)
gl.glEnable(gl.MULTISAMPLE)
gl.glClearColor(0,0,0,1)




-----------------------------------------------------------------------------
--[[                             RENDER LOOP                             ]]--
-----------------------------------------------------------------------------


local function getFramebufferSize(window)
    local pWidth = ffi.new 'int[1]'
    local pHeight = ffi.new 'int[1]'
    gl.glfwGetFramebufferSize(window, pWidth, pHeight)
    return pWidth[0], pHeight[0]
end





-- Liszt Mumbo Jumbo

local time_global = L.NewGlobal(L.float, 0)
vertices:NewField('pos_temp', L.vector(L.float, 4)):Load({0,0,0,0})
local update_pos1 = liszt kernel (v : vertices)
    var y = v.pos[1]
    if y > 0 then
        y = L.float(0.5 + 0.4 * C.cos(3*time_global))
    end
    v.pos_temp = { v.pos[0], y, v.pos[2], v.pos[3] }
end
local update_pos2 = liszt kernel (v : vertices)
    var pos = v.pos_temp
    v.pos = pos
    vdb.color(v.color)
    vdb.point(pos)
end




-- loop
while gl.glfwWindowShouldClose(window) == 0 do
    -- render the spinning triangle

    local width, height = getFramebufferSize(window)
    local ratio = width / height
    gl.glViewport(0, 0, width, height)
    gl.glDepthRange(0, 1)

    gl.glClear( bit.bor( gl.COLOR_BUFFER_BIT, gl.DEPTH_BUFFER_BIT ))


    -- set the shader program to use
    shader:use()

    -- Set up the matrix
    local time = gl.glfwGetTime()
    local proj = mat4f.ortho(ratio, 1, 0, 1)
    local rotation = mat4f.rotz(time)
    shader:setUniform('pvm', proj * rotation)


    time_global:set(time)
    update_pos1(vertices)
    update_pos2(vertices)
    triangles:UpdateGClef()

    -- Issue the actual draw call
    triangles:DrawGClef()



    gl.glfwSwapBuffers(window)
    -- polling with JIT compilation will cause errors
    -- b/c the underlying C code will invoke the callbacks.
    -- See: http://luajit.org/ext_ffi_semantics.html#callback
    jit.off(gl.glfwPollEvents())
    -- glfwWaitEvents() is recommended
    -- when the user is not manipulating any UI.
    -- It is not an appropriate choice when the execution itself
    -- is causing updates, unless that execution spoofs events
    -- to glfw to force rendering updates
    --gl.glfwWaitEvents();
end



-- Cleanup
gl.glfwTerminate()
print 'safe exit'


