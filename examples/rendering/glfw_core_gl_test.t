
--[[

    This file builds on the basic GLFW test.
    It demonstrates how we can get a modern OpenGL Core context set up.

    It does not attempt to produce a useful factorization of OpenGL Core
    functionality.

]]--

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

-- set up the data to load
local n_faces = 1
local n_vertices = n_faces * 3
local vertex_positions = terralib.global(float[n_vertices * 4])
vertex_positions:set({
 -0.6, -0.4, 0, 1,
  0.6, -0.4, 0, 1,
    0,  0.6, 0, 1
})
local vertex_colors = terralib.global(float[n_vertices * 3])
vertex_colors:set({
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
})


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
         cosr/ratio,  sinr, 0, 0,
        -sinr/ratio,  cosr, 0, 0,
                  0,     0, 1, 0,
                  0,     0, 0, 1,
    })
    local pvm_location =
        glfw.glGetUniformLocation(shader_program, 'pvm')
    glfw.glUniformMatrix4fv(pvm_location,
        1, false, -- expect column-major matrices
        rot_ortho_matrix:get())


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


