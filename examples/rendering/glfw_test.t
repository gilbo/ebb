
--[[

    This file demonstrates a way to get GLFW set up using Terra.

    We use old-style OpenGL to keep this file focused on the problem
    of getting GLFW set up.

]]--

local ffi = require("ffi")
local glfw = terralib.includecstring([[
//#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

int GLFW_KEY_ESCAPE_val     () { return GLFW_KEY_ESCAPE     ; }
int GLFW_PRESS_val          () { return GLFW_PRESS          ; }
int GL_COLOR_BUFFER_BIT_val () { return GL_COLOR_BUFFER_BIT ; }
int GL_PROJECTION_val       () { return GL_PROJECTION       ; }
int GL_MODELVIEW_val        () { return GL_MODELVIEW        ; }
int GL_TRIANGLES_val        () { return GL_TRIANGLES        ; }

#ifdef __APPLE__
    char *CURRENT_PLATFORM_STR() { return "apple"; }
#else
    char *CURRENT_PLATFORM_STR() { return ""; }
#endif
]])
local platform = ffi.string(glfw.CURRENT_PLATFORM_STR());

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


-- Route GLFW errors to the console
glfw.glfwSetErrorCallback(function(err_code, err_str)
    io.stderr:write(err_str)
end)

-- init glfw
if not glfw.glfwInit() then
    error('failed to initialize GLFW')
end


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
    glfw.glClear(glfw.GL_COLOR_BUFFER_BIT_val())
    glfw.glMatrixMode(glfw.GL_PROJECTION_val());
    glfw.glLoadIdentity();
    glfw.glOrtho(-ratio, ratio, -1, 1, 1, -1);
    glfw.glMatrixMode(glfw.GL_MODELVIEW_val());
    glfw.glLoadIdentity();
    glfw.glRotatef(glfw.glfwGetTime() * 50, 0, 0, 1);
    glfw.glBegin(glfw.GL_TRIANGLES_val());
    glfw.glColor3f(1, 0, 0);
    glfw.glVertex3f(-0.6, -0.4, 0);
    glfw.glColor3f(0, 1, 0);
    glfw.glVertex3f(0.6, -0.4, 0);
    glfw.glColor3f(0, 0, 1);
    glfw.glVertex3f(0, 0.6, 0)
    glfw.glEnd()


    glfw.glfwSwapBuffers(window)
    glfw.glfwPollEvents()
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


