
--[[

    This file builds on the basic GLFW test.
    It demonstrates how we can get a modern OpenGL Core context set up.

    It does not attempt to produce a useful factorization of OpenGL Core
    functionality.

]]--

import "compiler.liszt"
LDB = terralib.require "compiler.ldb"
Grid = terralib.require "compiler.grid"
local print = L.print



local ffi = require("ffi")
local glfw = terralib.includecstring([[
//#define GLFW_INCLUDE_GLU
#define GLFW_INCLUDE_GLCOREARB
#include <GLFW/glfw3.h>

#include <stdlib.h> // for malloc below
#include <stdio.h>

int GLFW_KEY_ESCAPE_val     () { return GLFW_KEY_ESCAPE     ; }
int GLFW_PRESS_val          () { return GLFW_PRESS          ; }

void requestCoreContext() {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}

void loadId(GLint location) {
    float id[] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    glUniformMatrix4fv(location, 1, GL_FALSE, id);
}

#ifdef __APPLE__
    char *CURRENT_PLATFORM_STR() { return "apple"; }
#else
    char *CURRENT_PLATFORM_STR() { return ""; }
#endif
]])
local platform = ffi.string(glfw.CURRENT_PLATFORM_STR());
--for k,v in pairs(glfw) do print(k,v) end

local function map_gl_enums(const_list)
    local c_vals = terralib.cast(&glfw.GLenum,
        glfw.malloc(#const_list * terralib.sizeof(glfw.GLenum)))

    local func = [[
    #define GLFW_INCLUDE_GLCOREARB
    #include <GLFW/glfw3.h>
    void get_constants(GLenum *output) {
    ]]
    for i,name in ipairs(const_list) do
        func = func .. 
        'output['..tostring(i-1)..'] = '.. 'GL_' .. name .. ';\n'
    end
    func = func..'}'

    terralib.includecstring(func).get_constants(c_vals)

    local result = {}
    for i=1,#const_list do
        result[const_list[i]] = c_vals[i-1]
    end

    return result
end
-- GL C(onstants)
local GLC = map_gl_enums({
    'COLOR_BUFFER_BIT',
    'DEPTH_BUFFER_BIT',
    'TRIANGLES',
    'DEPTH_TEST',
    'CULL_FACE',
    'MULTISAMPLE',
    'ARRAY_BUFFER',
    'ELEMENT_ARRAY_BUFFER',
    'STATIC_DRAW',
    'FLOAT',
    'VERTEX_SHADER',
    'FRAGMENT_SHADER',
    'INFO_LOG_LENGTH',
    'VALIDATE_STATUS',
    'LINK_STATUS',
    'FALSE',
    'TRUE',
})



-- do platform specific library location and linking
if platform == 'apple' then
    local glfw_path = nil
    for line in io.popen('mdfind -name libglfw'):lines() do
        if glfw_path then
            error('when searching for libglfw, multiple options were found. '..
                  'Crashing as a safety measure.')
        end
        glfw_path = line
    end
    terralib.linklibrary(glfw_path)
else
    error('did not recognize platform')
end


-----------------------------------------------------------------------------
--[[                             GLFW SETUP                              ]]--
-----------------------------------------------------------------------------


-- Route GLFW errors to the console
glfw.glfwSetErrorCallback(function(err_code, err_str)
    io.stderr:write('GLFW ERROR; code: '..tostring(err_code)..'\n')
    io.stderr:write(ffi.string(err_str) .. '\n')
end)

-- init glfw
if not glfw.glfwInit() then
    error('failed to initialize GLFW')
end

-- request a GL Core (forward compatible) Context
-- Keep it simple to demonstrate ONLY GLFW here
glfw.requestCoreContext()



-- create the window
local window = glfw.glfwCreateWindow(640, 480, "Hello World", nil, nil);
if not window then
    glfw.glfwTerminate()
    error('Failed to create GLFW Window')
end
glfw.glfwMakeContextCurrent(window)

-- example input callback
-- Close on ESC
glfw.glfwSetKeyCallback(window,
function(window, key, scancode, action, mods)
    if key == glfw.GLFW_KEY_ESCAPE_val() and
       action == glfw.GLFW_PRESS_val()
    then
        glfw.glfwSetWindowShouldClose(window, true)
    end
end)


-----------------------------------------------------------------------------
--[[                               INIT VAO                              ]]--
-----------------------------------------------------------------------------

local function genVAO()
    local p_vao = ffi.new 'unsigned int[1]'
    glfw.glGenVertexArrays(1,p_vao)
    return p_vao[0]
end

local function genBuffers(n)
    local p_buf = ffi.new ('unsigned int['..tostring(n)..']')
    glfw.glGenBuffers(n, p_buf)
    local buffers = {}
    for i=0,n-1 do
        table.insert(buffers, p_buf[i])
    end
    return buffers
end

-- get a vertex array object and set it to be current
local vao = genVAO()
glfw.glBindVertexArray(vao)

-- allocate some buffers in the VAO
local buffers = genBuffers(2)

-- mjlgg: Dimension for fluid grid comes from here
local dim = {4, 4}

globals = {}
globals['viscosity'] = 0.00001
globals['diffusion'] = 0.001
dt = 0.008

local fluid = Grid.GridClass:initUniformGrid(dim, {1, 1, 1, 1, 1}, globals)
local fluidPrev = Grid.GridClass:initUniformGrid(dim, {1, 1, 1, 1, 1}, globals)

-- set up the data to load
local n_faces = (dim[1] - 1) * (dim[2] - 1) * 2
local n_vertices = n_faces * 3
local vertex_positions = terralib.global(float[n_vertices * 4])

vp = {}

offset = {-.4, -.4}
size = {.3, .4}

c = 0
s = .5
size = {size[1] * s * 0.5, size[2] * s * 0.5}

for i = 1, dim[1] - 1 do
    for j = 1, dim[2] - 1 do
        x = 0.3 * s * i
        y = 0.4 * s * j

        vp[c    ] = (x - size[1]) + offset[1]
        vp[c + 1] = (y - size[2]) + offset[2]
        vp[c + 2] = 0
        vp[c + 3] = 1

        vp[c + 4] = (x + size[1]) + offset[1]
        vp[c + 5] = (y - size[2]) + offset[2]
        vp[c + 6] = 0
        vp[c + 7] = 1

        vp[c + 8] = (x - size[1]) + offset[1]
        vp[c + 9] = (y + size[2]) + offset[2]
        vp[c + 10] = 0
        vp[c + 11] = 1

        vp[c + 12] = (x + size[1]) + offset[1]
        vp[c + 13] = (y + size[2]) + offset[2]
        vp[c + 14] = 0
        vp[c + 15] = 1

        vp[c + 16] = (x - size[1]) + offset[1]
        vp[c + 17] = (y + size[2]) + offset[2]
        vp[c + 18] = 0
        vp[c + 19] = 1

        vp[c + 20] = (x + size[1]) + offset[1]
        vp[c + 21] = (y - size[2]) + offset[2]
        vp[c + 22] = 0
        vp[c + 23] = 1

        c = c + 24
    end
end

vc = {}

c = 0

for i = 1, dim[1] - 1 do
    for j = 1, dim[2] - 1 do
        vc[c    ] = 1
        vc[c + 1] = 0
        vc[c + 2] = 0

        vc[c + 3] = 0
        vc[c + 4] = 1
        vc[c + 5] = 0

        vc[c + 6] = 0
        vc[c + 7] = 0
        vc[c + 8] = 1

        vc[c + 9] = 1
        vc[c + 10] = 0
        vc[c + 11] = 0

        vc[c + 12] = 0
        vc[c + 13] = 0
        vc[c + 14] = 1

        vc[c + 15] = 0
        vc[c + 16] = 1
        vc[c + 17] = 0
        
        c = c + 18
    end
end

--[[
vertex_positions:set({
 -0.5, -0.5, 0, 1,
  0.5, -0.5, 0, 1,
  0.5,  0.5, 0, 1,
})
]]

vertex_positions:set(vp)
local vertex_colors = terralib.global(float[n_vertices * 3])

vertex_colors:set(vc)

--[[
vertex_colors:set({
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
})
]]


-- Load Data
local pos_slot = 0
local pos_buffer = buffers[1]
glfw.glEnableVertexAttribArray(pos_slot)
glfw.glBindBuffer(GLC.ARRAY_BUFFER, pos_buffer)
glfw.glBufferData(GLC.ARRAY_BUFFER,
                  n_vertices * 4 * terralib.sizeof(float),
                  vertex_positions:getpointer(),
                  GLC.STATIC_DRAW)
-- says each element is 4 floats
-- false says do not notmalize vector,
-- 0 stride means tightly packed
-- final null is the offset to begin indexing at
glfw.glVertexAttribPointer(pos_slot, 4, GLC.FLOAT, false, 0, nil)

local color_slot = 1
local color_buffer = buffers[2]
glfw.glEnableVertexAttribArray(color_slot)
glfw.glBindBuffer(GLC.ARRAY_BUFFER, color_buffer)
glfw.glBufferData(GLC.ARRAY_BUFFER,
                  n_vertices * 3 * terralib.sizeof(float),
                  vertex_colors:getpointer(),
                  GLC.STATIC_DRAW)
glfw.glVertexAttribPointer(color_slot, 3, GLC.FLOAT, false, 0, nil)

-- index buffer
--glfw.glBindBuffer(GLC.ELEMENT_ARRAY_BUFFER, buffer_num_handle)
--glfw.glBufferData(GLC.ELEMENT_ARRAY_BUFFER, size, data_ptr, GLC.STATIC_DRAW)

-- clear any assigned vertex array object
glfw.glBindVertexArray(0)



-----------------------------------------------------------------------------
--[[                             INIT SHADER                             ]]--
-----------------------------------------------------------------------------

local vert_src = terralib.global(&int8,[[#version 330

//layout (std140) uniform Matrices {
//    mat4 pvm;
//} ;
uniform mat4 pvm;

in vec4 position;
in vec3 vcolor;

out vec4 color;

void main()
{
    color = vec4(vcolor,1.0);
    gl_Position = pvm * position ;
} 
]])

local frag_src = terralib.global(&int8,[[#version 330

in vec4 color;

out vec4 outputF;

void main()
{
    outputF = color;
} 
]])


-- VSShaderLib shader
local shader_program = glfw.glCreateProgram()

local vert_shader = glfw.glCreateShader(GLC.VERTEX_SHADER)
glfw.glShaderSource(vert_shader, 1, vert_src:getpointer(), nil)
glfw.glAttachShader(shader_program, vert_shader)
glfw.glCompileShader(vert_shader)

local frag_shader = glfw.glCreateShader(GLC.FRAGMENT_SHADER)
glfw.glShaderSource(frag_shader, 1, frag_src:getpointer(), nil)
glfw.glAttachShader(shader_program, frag_shader)
glfw.glCompileShader(frag_shader)


-- 0 is the colornumber saying which buffer rendering gets dumped to
-- 'outputF' is a string declaring what variable the output should be
-- written to in the Fragment Shader
glfw.glBindFragDataLocation(shader_program, 0, 'outputF')

glfw.glBindAttribLocation(shader_program, pos_slot, 'position')
glfw.glBindAttribLocation(shader_program, color_slot, 'vcolor')

glfw.glLinkProgram(shader_program)

-- bind vao to validate
glfw.glBindVertexArray(vao)
glfw.glValidateProgram(shader_program)
glfw.glBindVertexArray(0)


local getShaderInfo -- assigned function below
local getProgramInfo -- assigned function below
do
    local terra shader_info_helper(shader : glfw.GLuint) : &int8
        var buffer  : &int8 = nil
        var log_len : int
        glfw.glGetShaderiv(shader, GLC.INFO_LOG_LENGTH, &log_len)
        if log_len > 0 then
            buffer = [&int8](glfw.malloc(log_len))
            var chars_written : int
            glfw.glGetShaderInfoLog(shader, log_len, &chars_written, buffer)
            -- ignore chars_written for now???
        end
        return buffer
    end
    local terra program_info_helper(program : glfw.GLuint) : &int8
        var buffer  : &int8 = nil
        var log_len : int
        glfw.glGetProgramiv(program, GLC.INFO_LOG_LENGTH, &log_len)
        if log_len > 0 then
            buffer = [&int8](glfw.malloc(log_len))
            var chars_written : int
            glfw.glGetProgramInfoLog(program, log_len, &chars_written, buffer)
        end
        return buffer
    end
    local function getInfo(is_shader, obj)
        local buffer
        if is_shader then
            buffer = shader_info_helper(obj)
        else
            buffer = program_info_helper(obj)
        end

        if buffer ~= nil then -- must check nil b/c buffer is cdata
            local result = ffi.string(buffer)
            glfw.free(buffer)
            return result
        else
            return 'no info found'
        end
    end
    getShaderInfo = function(shader) return getInfo(true, shader) end
    getProgramInfo = function(program) return getInfo(false, program) end
end

print(
[[/--------------------\
|vert shader info log|
\--------------------/
]]..getShaderInfo(vert_shader))
print(
[[/--------------------\
|frag shader info log|
\--------------------/
]]..getShaderInfo(frag_shader))
print(
[[/-----------------------\
|shader program info log|
\-----------------------/
]]..getProgramInfo(shader_program))

print('')

--[[
1280 GL_INVALID_ENUM      
1281 GL_INVALID_VALUE     
1282 GL_INVALID_OPERATION 
1283 GL_STACK_OVERFLOW    
1284 GL_STACK_UNDERFLOW   
1285 GL_OUT_OF_MEMORY
]]--
print('gl error code (see source; 0 is all ok): ', glfw.glGetError())


print('shader program validity...')
local p_validate_program = ffi.new 'int[1]'
glfw.glGetProgramiv(shader_program, GLC.VALIDATE_STATUS, p_validate_program)
if p_validate_program[0] == 0 then
    print('invalid program')
else
    print('valid program')
end

-- glfw.glDeleteProgram(shader_program)
-- glfw.glDeleteShader(vert_shader)
-- glfw.glDeleteShader(frag_shader)


-----------------------------------------------------------------------------
--[[                             INIT OPENGL                             ]]--
-----------------------------------------------------------------------------


-- INIT OPENGL
glfw.glEnable(GLC.DEPTH_TEST)
glfw.glEnable(GLC.CULL_FACE)
glfw.glEnable(GLC.MULTISAMPLE)
glfw.glClearColor(0,0,0,1)


-----------------------------------------------------------------------------
--[[                             FLUID STUFF                             ]]--
-----------------------------------------------------------------------------

function addSource(dim, dstGrid, dstIndex, srcGrid, srcIndex, dt)
    for i = 1, dim[1] + 2 do
        for j = 1, dim[2] + 2 do
            local x = dstGrid:get({i, j})
            local y = srcGrid:get({i, j})
            local z = {x[1], x[2], x[3], x[4], x[5]}

            z[dstIndex] = x[dstIndex] + y[srcIndex] * dt

            dstGrid:set({i, j}, z)
        end
    end
end

function setBoundary(dim, grid, index, boundaryFlag)
    for i = 2, dim[1] + 1 do
        -- 1
        local x = grid:get({1, i})
        local y = grid:get({2, i})
        local z = {x[1], x[2], x[3], x[4], x[5]}

        if boundaryFlag == 1 then
            z[index] = -y[index]
        else
            z[index] = y[index]
        end

        grid:set({1, i}, z)

        -- 2
        local x = grid:get({dim[1] + 2, i})
        local y = grid:get({dim[1] + 1, i})
        local z = {x[1], x[2], x[3], x[4], x[5]}

        if boundaryFlag == 1 then
            z[index] = -y[index]
        else
            z[index] = y[index]
        end

        grid:set({dim[1] + 2, i}, z)

        -- 3
        local x = grid:get({i, 1})
        local y = grid:get({i, 2})
        local z = {x[1], x[2], x[3], x[4], x[5]}

        if boundaryFlag == 2 then
            z[index] = -y[index]
        else
            z[index] = y[index]
        end

        grid:set({i, 1}, z)

        -- 4
        local x = grid:get({i, dim[2] + 2})
        local y = grid:get({i, dim[2] + 1})
        local z = {x[1], x[2], x[3], x[4], x[5]}

        if boundaryFlag == 2 then
            z[index] = -y[index]
        else
            z[index] = y[index]
        end

        grid:set({1, i}, z)
    end

    -- 1
    local x = grid:get({1, 1})
    local y = grid:get({2, 1})
    local z = grid:get({1, 2})
    local w = {x[1], x[2], x[3], x[4], x[5]}

    w[index] = 0.5 * (y[index] + z[index])

    grid:set({1, 1}, w)

    -- 2
    local x = grid:get({1, dim[2] + 2})
    local y = grid:get({2, dim[2] + 2})
    local z = grid:get({1, dim[2] + 1})
    local w = {x[1], x[2], x[3], x[4], x[5]}

    w[index] = 0.5 * (y[index] + z[index])

    grid:set({1, dim[2] + 2}, w)

    -- 3
    local x = grid:get({dim[1] + 2, 1})
    local y = grid:get({dim[1] + 1, 2})
    local z = grid:get({dim[1] + 2, 2})
    local w = {x[1], x[2], x[3], x[4], x[5]}

    w[index] = 0.5 * (y[index] + z[index])

    grid:set({dim[1] + 2, 1}, w)

    -- 4
    local x = grid:get({dim[1] + 2, dim[2] + 2})
    local y = grid:get({dim[1] + 1, dim[2] + 2})
    local z = grid:get({dim[1] + 2, 2})
    local w = {x[1], x[2], x[3], x[4], x[5]}

    w[index] = 0.5 * (y[index] + z[index])

    grid:set({dim[1] + 2, dim[2] + 2}, w)
end

function advect(dim, dstGrid, dstIndex, srcGrid, srcIndex, uGrid, uIndex, vGrid, vIndex, dt, boundaryFlag)
    local dt0 = dt * (dim[1] + 1)

    for i = 2, dim[1] + 1 do
        for j = 2, dim[2] + 1 do
            -- x
            local u0 = uGrid:get({i, j})
            local x = i - dt0 * u0[uIndex]
 
            if x < 0.5 then
                x = 0.5
            end

            if x > dim[1] + 0.5 then
                x = dim[1] + 0.5
            end

            local i0 = math.floor(x)
            local i1 = i0 + 1

            -- y
            local v0 = vGrid:get({i, j})
            local y = j - dt0 * v0[vIndex]

            if y < 0.5 then
                y = 0.5
            end

            if y > dim[2] + 0.5 then
                y = dim[2] + 0.5
            end

            local j0 = math.floor(x)
            local j1 = j0 + 1

            -- ...
            local s1 = x - i0
            local s0 = 1 - s1
            local t1 = y - j0
            local t0 = 1 - t1

            local d = dstGrid:get({i, j})
            local d00 = srcGrid:get({i0, j0})
            local d01 = srcGrid:get({i0, j1})
            local d10 = srcGrid:get({i1, j0})
            local d11 = srcGrid:get({i1, j1})
            local z = {d[1], d[2], d[3], d[4], d[5]}

            z[dstIndex] = s0 * (t0 * d00[srcIndex] + t1 * d01[srcIndex]) + s1 * (t0 * d10[srcIndex] + t1 * d11[srcIndex])

            dstGrid:set({i, j}, z)
        end
    end

    setBoundary(dim, dstGrid, dstIndex, boundaryFlag)
end

function diffuse(dim, dstGrid, srcGrid, index, diff, dt, boundaryFlag)
    local a = dt * diff * (dim[1] + 1) * (dim[2] + 1)

    for k = 1, 20 do
        for i = 2, dim[1] + 1 do
            for j = 2, dim[2] + 1 do
                local x = dstGrid:get({i, j})
                local y = srcGrid:get({i, j})
                local z = {x[1], x[2], x[3], x[4], x[5]}

                local leftX = dstGrid:get({i - 1, j})
                local rightX = dstGrid:get({i + 1, j})
                local botX = dstGrid:get({i, j - 1})
                local topX = dstGrid:get({i, j + 1})

                z[index] = (y[index] + a * (leftX[index] + rightX[index] + botX[index] + topX[index])) / (1 + 4 * a)
                dstGrid:set({i, j}, z)
            end
        end

        setBoundary(dim, dstGrid, index, boundaryFlag)
    end
end

function project(dim, uGrid, uIndex, vGrid, vIndex, pGrid, pIndex, divGrid, divIndex)
    local h = 1 / (dim[1] + 1)

    for i = 2, dim[1] + 1 do
        for j = 2, dim[2] + 1 do
            local x = divGrid:get({i, j})
            local y = {x[1], x[2], x[3], x[4], x[5]}

            local leftU = uGrid:get({i - 1, j})
            local rightU = uGrid:get({i + 1, j})

            local botV = vGrid:get({i, j + 1})
            local topV = vGrid:get({i, j - 1})

            y[divIndex] = -0.5 * h * (rightU[uIndex] - leftU[uIndex] + botV[vIndex] - topV[vIndex])
            divGrid:set({i, j}, y)

            local z = pGrid:get({i, j})
            local w = {z[1], z[2], z[3], z[4], z[5]}

            w[pIndex] = 0
            pGrid:set({i, j}, w)
        end
    end

    setBoundary(dim, divGrid, divIndex, 0)
    setBoundary(dim, pGrid, pIndex, 0)

    for k = 1, 20 do
        for i = 2, dim[1] + 1 do
            for j = 2, dim[2] + 1 do
                local x = pGrid:get({i, j})
                local y = {x[1], x[2], x[3], x[4], x[5]}

                local div = divGrid:get({i, j})

                local leftP = pGrid:get({i - 1, j})
                local rightP = pGrid:get({i + 1, j})

                local botP = pGrid:get({i, j + 1})
                local topP = pGrid:get({i, j - 1})

                y[pIndex] = 0.25 * (div[divIndex] + rightP[pIndex] + leftP[pIndex] + botP[pIndex] + topP[pIndex])
                pGrid:set({i, j}, y)
            end
        end

        setBoundary(dim, pGrid, pIndex, 0)
    end

    for i = 2, dim[1] + 1 do
        for j = 2, dim[2] + 1 do
            local u = uGrid:get({i, j})
            local u0 = {u[1], u[2], u[3], u[4], u[5]}

            local leftP = pGrid:get({i - 1, j})
            local rightP = pGrid:get({i + 1, j})

            u0[uIndex] = (rightP[pIndex] - leftP[pIndex]) / h
            uGrid:set({i, j}, u0)

            local v = vGrid:get({i, j})
            local v0 = {v[1], v[2], v[3], v[4], v[5]}

            local botP = pGrid:get({i, j + 1})
            local topP = pGrid:get({i, j - 1})

            v0[vIndex] = (botP[pIndex] - topP[pIndex]) / h
            vGrid:set({i, j}, v0)
        end
    end

    setBoundary(dim, uGrid, uIndex, 1)
    setBoundary(dim, vGrid, vIndex, 2)
end

function dens_step(dim, dstGrid, dstIndex, srcGrid, srcIndex, uGrid, uIndex, vGrid, vIndex, diff, dt)
    assert(dstIndex == srcIndex)
    addSource(dim, dstGrid, dstIndex, srcGrid, srcIndex, dt)
    diffuse(dim, dstGrid, srcGrid, srcIndex, diff, dt, 0)
    advect(dim, dstGrid, dstIndex, srcGrid, srcIndex, uGrid, uIndex, vGrid, vIndex, dt, 0)
end

function vel_step(dim, uDstGrid, vDstGrid, uSrcGrid, vSrcGrid, uIndex, vIndex, visc, dt)
    addSource(dim, uDstGrid, uIndex, uSrcGrid, uIndex, dt)
    addSource(dim, vDstGrid, vIndex, vSrcGrid, vIndex, dt)

    diffuse(dim, uSrcGrid, uDstGrid, uIndex, visc, dt, 1)
    diffuse(dim, vSrcGrid, vDstGrid, vIndex, visc, dt, 2)

    project(dim, uDstGrid, uIndex, vDstGrid, vIndex, uSrcGrid, uIndex, vSrcGrid, vIndex)
    advect(dim, uSrcGrid, uIndex, uDstGrid, uIndex, vSrcGrid, vIndex, uSrcGrid, uIndex, dt, 1)
    advect(dim, vSrcGrid, vIndex, vDstGrid, vIndex, vSrcGrid, vIndex, uSrcGrid, uIndex, dt, 2)
    project(dim, uDstGrid, uIndex, vDstGrid, vIndex, uSrcGrid, uIndex, vSrcGrid, vIndex)
end


-----------------------------------------------------------------------------
--[[                             RENDER LOOP                             ]]--
-----------------------------------------------------------------------------


local function getFramebufferSize(window)
    local pWidth = ffi.new 'int[1]'
    local pHeight = ffi.new 'int[1]'
    glfw.glfwGetFramebufferSize(window, pWidth, pHeight)
    return pWidth[0], pHeight[0]
end


-- loop
while glfw.glfwWindowShouldClose(window) == 0 do
    -- render the spinning triangle

    local width, height = getFramebufferSize(window)
    local ratio = width / height
    glfw.glViewport(0, 0, width, height)

    glfw.glClear( bit.bor( GLC.COLOR_BUFFER_BIT, GLC.DEPTH_BUFFER_BIT ))


    -- set the shader program to use
    glfw.glUseProgram(shader_program)


    -- Do the matrix stuff manually now
    local rad = glfw.glfwGetTime()
    local sinr = math.sin(rad)
    local cosr = math.cos(rad)

    local identity_matrix = terralib.global(float[16])
    identity_matrix:set({
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    })
    local rot_ortho_matrix = terralib.global(float[16])
    rot_ortho_matrix:set({
--         cosr/ratio,  sinr, 0, 0,
--        -sinr/ratio,  cosr, 0, 0,
                  1,     0, 0, 0,
                  0,     1, 0, 0,
                  0,     0, 1, 0,
                  0,     0, 0, 1,
    })
    local pvm_location =
        glfw.glGetUniformLocation(shader_program, 'pvm')
    glfw.glUniformMatrix4fv(pvm_location,
        1, false, -- expect column-major matrices
        rot_ortho_matrix:get())

    -- Fluids
    print("Running vel_step()...")
    --vel_step({2, 2}, fluid, fluidPrev, fluid, fluidPrev, 1, 2, fluid['globals']['viscosity'], dt)
    
    print("Running dens_step()...")
    --dens_step({2, 2}, fluid, 5, fluidPrev, 5, fluid, 1, fluid, 2, fluid['globals']['diffusion'], dt)
    
    -- Update density
    for i = 1, 2 do
        for j = 1, 2 do
            c = 18 * (i - 1) + 18 * (j - 1) * (dim[1] - 1)
            lowc = 18 * (i - 1) + 18 * (j - 2) * (dim[1] - 1)
            d = fluid[i][j][5]
            
            if (i > 1) and (j < dim[2]) then
                -- 1
                vc[c - 18 + 3] = d
                vc[c - 18 + 4] = d
                vc[c - 18 + 5] = d
            
                -- 2
                vc[c - 18 + 15] = d
                vc[c - 18 + 16] = d
                vc[c - 18 + 17] = d
            end

            if (i < dim[1]) and (j < dim[2]) then
                -- 3 (center)
                vc[c] = d
                vc[c + 1] = d
                vc[c + 2] = d
            end

            if (i > 1) and (j > 1) then
                -- 4
                vc[lowc - 18 + 9] = d
                vc[lowc - 18 + 10] = d
                vc[lowc - 18 + 11] = d
            end

            if (i < dim[1]) and (j > 1) then
                -- 5
                vc[lowc + 6] = d
                vc[lowc + 7] = d
                vc[lowc + 8] = d

                -- 6
                vc[lowc + 12] = d
                vc[lowc + 13] = d
                vc[lowc + 14] = d
            end
        end
    end

    vertex_colors:set(vc)

    glfw.glEnableVertexAttribArray(color_slot)
    glfw.glBindBuffer(GLC.ARRAY_BUFFER, color_buffer)
    glfw.glBufferData(GLC.ARRAY_BUFFER,
                  n_vertices * 3 * terralib.sizeof(float),
                  vertex_colors:getpointer(),
                  GLC.STATIC_DRAW)
    glfw.glVertexAttribPointer(color_slot, 3, GLC.FLOAT, false, 0, nil)

    -- Issue the actual draw call
    glfw.glBindVertexArray(vao)
    glfw.glDrawArrays(GLC.TRIANGLES, 0, n_vertices)


    glfw.glfwSwapBuffers(window)
    -- polling with JIT compilation will cause errors
    -- b/c the underlying C code will invoke the callbacks.
    -- See: http://luajit.org/ext_ffi_semantics.html#callback
    jit.off(glfw.glfwPollEvents())
    -- glfwWaitEvents() is recommended
    -- when the user is not manipulating any UI.
    -- It is not an appropriate choice when the execution itself
    -- is causing updates, unless that execution spoofs events
    -- to glfw to force rendering updates
    --glfw.glfwWaitEvents();
end



-- Cleanup
glfw.glfwTerminate()
print 'safe exit'


