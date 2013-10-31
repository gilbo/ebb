local types   = terralib.require('compiler/types')
local Type    = types.Type
local t       = types.t

local function make_prototype(tb)
   tb.__index = tb
   return tb
end

local Macro = make_prototype {kind=Type.kinds.functype}
local MacroMT = {__index=Macro}

function Macro.new(check, codegen)
    return setmetatable({check=check, codegen=codegen}, MacroMT)
end

---------------------------------------------
--[[ Builtin macros                      ]]--
---------------------------------------------
function AssertCheck(ast, ctxt)
    local args = ast.params.children
    if #args ~= 1 then
        ctxt:error(ast, "assert expects exactly 1 argument (instead got " .. #args .. ")")
        return
    end

    local test = args[1]
    local test_type = test:check(ctxt).node_type
    if test_type ~= t.error and test_type ~= t.bool then
        ctxt:error(ast, "expected a boolean as the test for assert statement")
    end
end

local c = terralib.includecstring([[
#include <stdio.h>
#include <stdlib.h>
]])
local terra lisztAssert(test : bool, file : rawstring, line : int)
    if not test then
        c.printf("%s:%d: assertion failed!\n", file, line)
        c.exit(1)
    end
end

function AssertCodegen(ast, env)
    local test = ast.params.children[1]:codegen(env)
    return quote lisztAssert(test, ast.filename, ast.linenumber) end
end

function PrintCheck(ast, ctxt)
    local args = ast.params.children
    if #args ~= 1 then
        ctxt:error(ast, "print expects exactly one argument (instead got " .. #args .. ")")
    end

    local output = args[1]
    local outtype = output:check(ctxt).node_type
    if outtype ~= t.error and not outtype:isExpressionType() then
        ctxt:error(ast, "only numbers, bools, and vectors can be printed")
    end
end

function PrintCodegen(ast, env)
    local output = ast.params.children[1]
	local lt   = output.node_type
    local tt   = lt:terraType()
    local code = output:codegen(env)
    if     lt == t.float then return quote c.printf("%f\n", [float](code)) end
	elseif lt == t.int   then return quote c.printf("%d\n", code) end
	elseif lt == t.bool  then
        return quote c.printf("%s", terralib.select(code, "true\n", "false\n")) end
	elseif lt:isVector() then
        local printSpec = "{"
        local sym = symbol()
        local elemQuotes = {}
        local bt = lt:baseType()
        for i = 0, lt.N - 1 do
            if bt == t.float then
                printSpec = printSpec .. " %f"
                table.insert(elemQuotes, `[float](sym[i]))
            elseif bt == t.int then
                printSpec = printSpec .. " %d"
                table.insert(elemQuotes, `sym[i])
            elseif bt == t.bool then
                printSpec = printSpec .. " %s"
                table.insert(elemQuotes, `terralib.select(sym[i], "true", "false"))
            end
        end
        printSpec = printSpec .. " }\n"
        return quote
            var [sym] : tt = code
        in
            c.printf(printSpec, elemQuotes)
        end
    else
        assert(false and "printed object should always be number, bool, or vector")
    end
end

function DotCheck(ast, ctxt)
    local args = ast.params.children
    if #args ~= 2 then
        ctxt:error(ast, "dot expects exactly 2 arguments (instead got " .. #args .. ")")
        return
    end
    local lt1 = args[1]:check(ctxt).node_type
    local lt2 = args[2]:check(ctxt).node_type
    if not lt1:isVector() then
        ctxt:error(args[1], "first argument to dot must be a vector")
    end
    if not lt2:isVector() then
        ctxt:error(args[2], "second argument to dot must be a vector")
    end
    if lt1.N ~= lt2.N then
        ctxt:error(ast, "cannot dot vectors of differing lengths (" ..
                lt1.N .. " and " .. lt2.N .. ")")
    end
    if lt1:baseType() == t.bool then
        ctxt:error(args[1], "boolean vector passed as first argument to dot")
    end
    if lt2:baseType() == t.bool then 
        ctxt:error(args[2], "boolean vector passed as second argument to dot")
    end
    return types.type_meet(lt1:baseType(), lt2:baseType())
end

function DotCodegen(ast, env)
    local args = ast.params.children
    local v1 = symbol()
    local v2 = symbol()
    local N = args[1].node_type.N
    if N == 0 then
        return `0
    end
    local result = `v1[0] * v2[0]
    -- TODO: make this codegen a Terra for loop for super-long vectors
    for i = 1, N - 1 do
        result = `result + v1[i] * v2[i]
    end

    local tt1 = args[1].node_type:terraType()
    local tt2 = args[2].node_type:terraType()
    local code1 = args[1]:codegen(env)
    local code2 = args[2]:codegen(env)
    return quote
        var [v1] : tt1 = code1
        var [v2] : tt2 = code2
    in
        result
    end
end

L.print = Macro.new(PrintCheck, PrintCodegen)
L.assert = Macro.new(AssertCheck, AssertCodegen)
L.dot = Macro.new(DotCheck, DotCodegen)
