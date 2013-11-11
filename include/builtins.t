local types   = terralib.require('compiler/types')
local Type    = types.Type
local t       = types.t

local function make_prototype(tb)
   tb.__index = tb
   return tb
end

local Func = make_prototype {kind=Type.kinds.functype}

function Func.new(luafunc)
    local check = function(ast, ctxt)
        error('unimplemented builtin function typechecking')
    end
    local codegen = function(ast, env)
        error('unimplemented builtin function codegen')
    end
    local call = nil
    if luafunc then
        call = (function(self, ...) return luafunc(...) end)
    end
    return setmetatable({check=check, codegen=codegen},
                        {__index=Func, __call=call})
end

---------------------------------------------
--[[ Builtin functions                   ]]--
---------------------------------------------
local B = {}
function B.is_function(obj)
    return getmetatable(obj) and getmetatable(obj).__index == Func
end

B.assert = Func.new(assert)

function B.assert.check(ast, ctxt)
    local args = ast.params.children
    if #args ~= 1 then
        ctxt:error(ast, "assert expects exactly 1 argument (instead got " .. #args .. ")")
        return
    end

    local test = args[1]
    local test_type = test.node_type
    if test_type:isVector() then test_type = test_type:baseType() end
    if test_type ~= t.error and test_type ~= t.bool then
        ctxt:error(ast, "expected a boolean or vector of booleans as the test for assert statement")
    end
end

local c = terralib.includecstring([[
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

FILE *get_stderr () { return stderr; }
]])

local terra lisztAssert(test : bool, file : rawstring, line : int)
    if not test then
        c.fprintf(c.get_stderr(), "%s:%d: assertion failed!\n", file, line)
        c.exit(1)
    end
end

function B.assert.codegen(ast, env)
    local test = ast.params.children[1]
    local code = test:codegen(env)
    if test.node_type:isVector() then
        if test.node_type.N == 0 then return quote end end
        local tt = test.node_type:terraType()
        local v = symbol()
        local alltest = `v[0]
        for i = 1, test.node_type.N - 1 do
            alltest = `alltest and v[i]
        end
        return quote var [v] = code in lisztAssert(alltest, ast.filename, ast.linenumber) end
    else
        return quote lisztAssert(code, ast.filename, ast.linenumber) end
    end
end


B.print = Func.new(print)

function B.print.check(ast, ctxt)
    local args = ast.params.children
    if #args ~= 1 then
        ctxt:error(ast, "print expects exactly one argument (instead got " .. #args .. ")")
    end

    local output = args[1]
    local outtype = output.node_type
    if outtype ~= t.error and not outtype:isExpressionType() and not outtype:isRow() then
        ctxt:error(ast, "only numbers, bools, vectors and rows can be printed")
    end
end

function B.print.codegen(ast, env)
    local output = ast.params.children[1]
	local lt   = output.node_type
    local tt   = lt:terraType()
    local code = output:codegen(env)
    if     lt == t.float or lt == t.double then return quote c.printf("%f\n", [double](code)) end
	elseif lt == t.int   or lt == t.row    then return quote c.printf("%d\n", code) end
	elseif lt == t.bool  then
        return quote c.printf("%s", terralib.select(code, "true\n", "false\n")) end
	elseif lt:isVector() then
        local printSpec = "{"
        local sym = symbol()
        local elemQuotes = {}
        local bt = lt:baseType()
        for i = 0, lt.N - 1 do
            if bt == t.float or bt == t.double then
                printSpec = printSpec .. " %f"
                table.insert(elemQuotes, `[double](sym[i]))
            elseif bt == t.int then
                printSpec = printSpec .. " %d"
                table.insert(elemQuotes, `sym[i])
            elseif bt == t.bool then
                printSpec = printSpec .. " %s"
                table.insert(elemQuotes, `terralib.select(sym[i], "true", "false"))
            else
                error('Unrecognized type in print: ' .. tostring(bt:terraType()))
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


local function dot(a, b)
    if not a.type:isVector() then
        error("first argument to dot must be a vector", 2)
    end
    if not b.type:isVector() then
        error("second argument to dot must be a vector", 2)
    end
    if #a.data ~= #b.data then
        error("cannot dot vectors of differing lengths (" ..
                #a.data .. " and " .. #b.data .. ")")
    end
    local sum = 0
    for i = 1, #a.data do
        sum = sum + a.data[i] * b.data[i]
    end
    return sum
end

B.dot = Func.new(dot)

function B.dot.check(ast, ctxt)
    local args = ast.params.children
    if #args ~= 2 then
        ctxt:error(ast, "dot expects exactly 2 arguments (instead got " .. #args .. ")")
        return
    end
    local lt1 = args[1].node_type
    local lt2 = args[2].node_type
    if not lt1:isVector() then
        ctxt:error(args[1], "first argument to dot must be a vector")
    end
    if not lt2:isVector() then
        ctxt:error(args[2], "second argument to dot must be a vector")
    end
    if not lt1:isVector() or not lt2:isVector() then return end
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

function B.dot.codegen(ast, env)
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


local function cross(a, b)
    if not a.type:isVector() then
        error("first argument to cross must be a vector", 2)
    end
    if not b.type:isVector() then
        error("second argument to cross must be a vector", 2)
    end
    local av = a.data
    if #av ~= 3 then
        error("arguments to cross must be vectors of length 3 (first argument is length " ..
                #av .. ")")
    end
    local bv = b.data
    if #bv ~= 3 then
        error("arguments to cross must be vectors of length 3 (second argument is length " ..
                #bv .. ")")
    end
    return L.NewVector(types.type_meet(a.type:baseType(), b.type:baseType()), {
        av[2] * bv[3] - av[3] * bv[2],
        av[3] * bv[1] - av[1] * bv[3],
        av[1] * bv[2] - av[2] * bv[1]
    })
end 

B.cross = Func.new(cross)

function B.cross.check(ast, ctxt)
    local args = ast.params.children
    if #args ~= 2 then
        ctxt:error(ast, "cross expects exactly 2 arguments (instead got " .. #args .. ")")
        return
    end
    local lt1 = args[1].node_type
    local lt2 = args[2].node_type
    if not lt1:isVector() then
        ctxt:error(args[1], "first argument to cross must be a vector")
    end
    if not lt2:isVector() then
        ctxt:error(args[2], "second argument to cross must be a vector")
    end
    if not lt1:isVector() or not lt2:isVector() then return end
    if lt1.N ~= 3 then
        ctxt:error(ast, "arguments to cross must be vectors of length 3 (first argument is of length" ..  lt1.N .. ")")
    end
    if lt2.N ~= 3 then
        ctxt:error(ast, "arguments to cross must be vectors of length 3 (second argument is of length" ..  lt1.N .. ")")
    end
    if lt1:baseType() == t.bool then
        ctxt:error(args[1], "boolean vector passed as first argument to cross")
    end
    if lt2:baseType() == t.bool then 
        ctxt:error(args[2], "boolean vector passed as second argument to cross")
    end
    return types.type_meet(lt1, lt2)
end

function B.cross.codegen(ast, env)
    local args = ast.params.children

    local tt1 = args[1].node_type:terraType()
    local tt2 = args[2].node_type:terraType()
    local v1 = symbol(tt1)
    local v2 = symbol(tt2)

    local tp = types.type_meet(args[1].node_type:baseType(), args[2].node_type:baseType()):terraType()
    local code1 = args[1]:codegen(env)
    local code2 = args[2]:codegen(env)
    return quote
        var [v1] = code1
        var [v2] = code2
    in
        vectorof(tp, v1[1] * v2[2] - v1[2] * v2[1],
                     v1[2] * v2[0] - v1[0] * v2[2],
                     v1[0] * v2[1] - v1[1] * v2[0])
    end
end


local function length(v)
    if not v.type:isVector() then
        error("argument to length must be a vector", 2)
    end
    return c.sqrt(dot(v, v))
end

B.length = Func.new(length)

function B.length.check(ast, ctxt)
    local args = ast.params.children
    if #args ~= 1 then
        ctxt:error(ast, "length expects exactly 1 argument (instead got " .. #args .. ")")
        return t.error
    end
    local lt = args[1].node_type
    if not lt:isVector() then
        ctxt:error(args[1], "argument to length must be a vector")
        return t.error
    end
    if lt:baseType() == t.bool then
        ctxt:error(args[1], "boolean vector passed as argument to length")
    end
    return t.float
end

function B.length.codegen(ast, env)
    local args = ast.params.children
    local sq = symbol()
    local N = args[1].node_type.N
    if N == 0 then
        return `0
    end
    local len2 = `sq[0]
    -- TODO: make this codegen a Terra for loop for super-long vectors
    for i = 1, N - 1 do
        len2 = `len2 + sq[i]
    end

    local tt = args[1].node_type:terraType()
    local code = args[1]:codegen(env)
    return quote
        var v : tt = code
        var [sq] : tt = v * v
    in
        c.sqrt(len2)
    end
end


local function map(fn, list)
    local result = {}
    for i = 1, #list do
        table.insert(result, fn(list[i]))
    end
    return result
end

local function GetTypedSymbol(arg)
    return symbol(arg.node_type:terraType())
end

local function TerraCheck(func)
    func:compile()
    return function (ast, ctxt)
        local args = ast.params.children 
        local argsyms = map(GetTypedSymbol, args)
        local rettype = nil
        local try = function()
            local terrafunc = terra([argsyms]) return func([argsyms]) end
            terrafunc:compile()
            rettype = terrafunc:getdefinitions()[1]:gettype().returns
        end
        local success, retval = pcall(try)
        if not success then
            ctxt:error(ast, "couldn't fit parameters to signature of terra function")
            ctxt:error(ast, retval)
            return t.error
        end
        if #rettype > 1 then
            ctxt:error(ast, "terra function returns more than one value")
            return t.error
        end
        if #rettype == 1 and not types.terraToLisztType(rettype[1]) then
            ctxt:error(ast, "unable to use return type '" .. tostring(rettype[1]) .. "' of terra function in Liszt")
            return t.error
        end
        return #rettype == 0 and t.error or types.terraToLisztType(rettype[1])
    end
end

local function TerraCodegen(func)
    return function (ast, env)
        local args = ast.params.children 
        local argsyms = map(GetTypedSymbol, args)
        local init_params = quote end
        for i = 1, #args do
            local code = args[i]:codegen(env)
            init_params = quote
                init_params
                var [argsyms[i]] = code
            end
        end
        return quote init_params in func(argsyms) end
    end 
end


function B.terra_to_func(terrafn)
    local newfunc = Func.new()
    newfunc.check, newfunc.codegen = TerraCheck(terrafn), TerraCodegen(terrafn)
    return newfunc
end

B.addBuiltinsToNamespace = function (L)
    L.print  = B.print
    L.assert = B.assert
    L.dot    = B.dot
    L.cross  = B.cross
    L.length = B.length
    L.is_function = B.is_function
end

return B
