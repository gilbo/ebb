#!/usr/bin/luajit
local ffi = require("ffi")

local lscmd
if ffi.os == "Windows" then
    lscmd = "cmd /c dir /b /s"
else
    lscmd = "find . | cut -c 3-"
end

local passed     = {}
local bad_passed = {}
local failed     = {}
local disabled   = {}

local exclude = {
    ['tests/test.lua'] = true,
}

local disable_str = '--DISABLE-TEST'

local function is_disabled (filename)
    local h = io.open(filename, "r")
    local line = h:read()
    io.close(h)
    return line and string.sub(line,1,#disable_str) == disable_str
end

print("==================")
print("= Running tests...")
print("==================")
for line in io.popen(lscmd):lines() do
    if ffi.os == "Windows" then
        local cwd = io.popen("cmd /c echo %cd%"):read()
        line = line:sub(cwd:len()+2)
        line = line:gsub("\\","/")
    end
    local file = line:match("^(tests/.*%.t)$") or line:match("^(tests/.*%.lua)$")
    if file and not exclude[file] then
        if is_disabled(file) then
            table.insert(disabled, file)
        else
            print(file)
            local execstring = "./liszt " .. file
            --things in the fail directory should cause terra compiler errors
            --we dont check for the particular error
            --but we do check to see that the "Errors reported.." message prints
            --which suggests that the error was reported gracefully
            --(if the compiler bites it before it finishes typechecking then it will not print this)
            local success = os.execute(execstring)
            local should_fail = (file:match("fails/") ~= nil)
            if success ~= 0 and not should_fail then
                table.insert(failed,file)
            elseif success == 0 and should_fail then
                table.insert(bad_passed,file)
            else
                table.insert(passed,file)
            end
        end
    end
end
print("==================")
print()

local function printtests(nm,lst)
    if #lst > 0 then
        print("==================")
        print("= "..nm)
        print("==================")
        for i,e in ipairs(lst) do
            print(e)
        end
        print("==================")
        print()
    end
end
--printtests("passing tests",passed)
printtests("FAILING tests",failed)
printtests("passed but should have failed",bad_passed)
printtests("disabled tests",disabled)

print(tostring(#passed).." tests passed, "..tostring(#failed + #bad_passed).." tests failed. " .. tostring(#disabled) .. " tests disabled.")
