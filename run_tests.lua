#!./terra/terra
local ffi = require "ffi"

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

local GPU_disabled = not terralib.cudacompile

local GPU_test_str = '--GPU-TEST'
local function is_gpu_test (filename)
    local h = io.open(filename, "r")
    local line = h:read()
    io.close(h)
    return line and string.sub(line,1,#GPU_test_str) == GPU_test_str
end

local function output_name (filename)
    local outname = filename:gsub("/(.-)%.t$", "/%1.out")
    -- check whether the file exists
    if outname ~= filename then
        local f = io.open(outname,"r")
        if f then
            io.close(f)
            return outname
        end
    end
    -- implicitly return nil if there is no file match
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
    local out_file = file and output_name(file)
    if file and not exclude[file] then
        if is_disabled(file) then
            table.insert(disabled, file)
        elseif GPU_disabled and is_gpu_test(file) then
            table.insert(disabled, file)
        else
            print(file)
            local should_fail = (file:match("fails/") ~= nil)
            local execstring = "./liszt " .. file
            -- If we expect output from this test, log stdout
            if out_file then
                execstring = execstring .. " > .test_out"
            elseif should_fail then
                execstring = execstring .. " > /dev/null 2>&1"
            end

            --things in the fail directory should cause terra compiler errors
            --we dont check for the particular error
            local success = os.execute(execstring)
            -- if we expect output, modulate the success appropriately
            if out_file and success == 0 then
                -- compare .test_out to out_file
                diff_string = 'diff .test_out ' .. out_file
                success = os.execute(diff_string)
            end
            -- record/report failure/success appropriately
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

-- cleanup temp files if they exist
os.execute('rm .test_out')

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
