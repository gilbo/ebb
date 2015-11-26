local B = {}
package.loaded["ebb.src.builtins"] = B

local L = require "ebblib"
local T = require "ebb.src.types"
local C = require "ebb.src.c"
local G = require "ebb.src.gpu_util"
local AST = require "ebb.src.ast"


local errorT    = T.error
local floatT    = T.float
local doubleT   = T.double
local intT      = T.int
local uint64T   = T.uint64
local boolT     = T.bool

local keyT      = T.key
local vectorT   = T.vector
--local matrixT   = T.matrix

local CPU       = L.CPU
local GPU       = L.GPU

local is_relation   = L.is_relation

---------------------------------------------
--[[ Builtin functions                   ]]--
---------------------------------------------
local Builtin = {}
Builtin.__index = Builtin
Builtin.__call = function(self,...)
    return self.luafunc(...)
end

function Builtin.new(luafunc)
    local check = function(ast, ctxt)
        error('unimplemented builtin function typechecking')
    end
    local codegen = function(ast, ctxt)
        error('unimplemented builtin function codegen')
    end
    if not luafunc then
        luafunc = function() error("Cannot call from Lua code") end
    end
    return setmetatable({check=check, codegen=codegen, luafunc = luafunc},
                        Builtin)
end
function B.is_builtin(f)
    return getmetatable(f) == Builtin
end


-- internal macro style built-in
B.Where = Builtin.new()
function B.Where.check(call_ast, ctxt)
    if #(call_ast.params) ~= 2 then
        ctxt:error(ast, "Where expects exactly 2 arguments")
        return errorT
    end

    local w = AST.Where:DeriveFrom(call_ast)
    w.field = call_ast.params[1]
    w.key   = call_ast.params[2]
    return w:check(ctxt)
end


local function id_checks(fname, ast, ctxt, args)
    if #args ~= 1 then 
        ctxt:error(ast, fname.." expects exactly 1 argument (instead got " ..
                        tostring(#args) .. ")")
        return false
    end

    if not args[1].node_type:isscalarkey() then
        ctxt:error(ast, "expected a relational key as the argument for "..
                        fname.."()")
        return false
    end
    return true
end

B.id = Builtin.new()
function B.id.check(ast, ctxt)
    local args = ast.params

    if not id_checks('id', ast, ctxt, args) then return errorT end
    if args[1].node_type.ndims ~= 1 then
        ctxt:error(ast, "Can only use built-in id() on keys of "..
                        "non-grid relations; "..
                        "try using xid(), yid() or zid() instead.")
        return errorT
    end

    return uint64T
end
function B.id.codegen(ast, ctxt)
    return `[uint64]([ast.params[1]:codegen(ctxt)].a0)
end

B.xid = Builtin.new()
function B.xid.check(ast, ctxt)
    local args = ast.params

    if not id_checks('xid', ast, ctxt, args) then return errorT end
    if args[1].node_type.ndims == 1 then
        ctxt:error(ast, "Can only use built-in xid() on keys of "..
                        "grid relations; try using id() instead.")
        return errorT
    end

    return uint64T
end
function B.xid.codegen(ast, ctxt)
    return `[ast.params[1]:codegen(ctxt)].a0
end

B.yid = Builtin.new()
function B.yid.check(ast, ctxt)
    local args = ast.params

    if not id_checks('yid', ast, ctxt, args) then return errorT end
    if args[1].node_type.ndims == 1 then
        ctxt:error(ast, "Can only use built-in yid() on keys of "..
                        "grid relations; try using id() instead.")
        return errorT
    end

    return uint64T
end
function B.yid.codegen(ast, ctxt)
    return `[ast.params[1]:codegen(ctxt)].a1
end

B.zid = Builtin.new()
function B.zid.check(ast, ctxt)
    local args = ast.params

    if not id_checks('zid', ast, ctxt, args) then return errorT end
    if args[1].node_type.ndims == 1 then
        ctxt:error(ast, "Can only use built-in zid() on keys of "..
                        "grid relations; try using id() instead.")
        return errorT
    end
    if args[1].node_type.ndims < 3 then
        ctxt:error(ast, "The key argument to zid() refers to a 2d grid, "..
                        "so zid() doesn't make any sense.")
        return errorT
    end

    return uint64T
end
function B.zid.codegen(ast, ctxt)
    return `[ast.params[1]:codegen(ctxt)].a2
end


B.Affine = Builtin.new()
function B.Affine.check(ast, ctxt)
    local args = ast.params

    if #args ~= 3 then
        ctxt:error(ast,'Affine expects 3 arguments')
        return errorT
    end
    local dst_rel_arg   = args[1]
    local matrix        = args[2]
    local key_arg       = args[3]
    local ret_type      = nil

    -- check that the first and last arg are actually relations
    if not dst_rel_arg.node_type:isinternal() or
       not is_relation(dst_rel_arg.node_type.value)
    then
        ctxt:error(ast[1], "Affine expects a relation as the 1st argument")
        return errorT
    end
    if not key_arg.node_type:isscalarkey() then
        ctxt:error(ast[3], "Affine expects a key as the 3rd argument")
        return errorT
    end

    -- get the source and destination relations and check that they're grids
    local dst_rel = dst_rel_arg.node_type.value
    local src_rel = key_arg.node_type.relation
    if not dst_rel:isGrid() then
        ctxt:error(ast[1],
            "Affine expects a grid relation as the 1st argument")
        return errorT
    end
    if not src_rel:isGrid() then
        ctxt:error(ast[3], "Affine expects a grid key as the 3rd argument")
        return errorT
    end

    -- get dimensions out
    local dst_dims = dst_rel:Dims()
    local src_dims = src_rel:Dims()

    -- now check the matrix argument type
    if not matrix.node_type:ismatrix() or
       matrix.node_type.Nrow ~= #dst_dims or
       matrix.node_type.Ncol ~= #src_dims + 1
    then
        ctxt:error(ast[2], "Affine expects a matrix as the 2nd argument "..
            "with matching dimensions (needs to be "..
            tostring(#dst_dims).."-by-"..tostring(#src_dims + 1))
        return errorT
    end
    --if not matrix.node_type:isintegral() then
    --    ctxt:error(ast[2], "Affine expects a matrix of integral values")
    --    return errorT
    --end
    if not matrix.node_type:isnumeric() then
        ctxt:error(ast[2], "Affine expects a matrix of numeric values")
        return errorT
    end
    -- WE NEED TO CHECK CONST-NESS, but this seems to be
    -- the wrong time to do it
    --if not matrix:is(AST.MatrixLiteral) then
    --    ctxt:error(ast[2], "Compiler could not verify that "..
    --        "the matrix argument (2nd) to Affine is constant")
    --    return errorT
    --end
    --for yi = 0,matrix.n-1 do for xi = 0,matrix.m-1 do
    --    if not matrix.elems[yi*matrix.m + xi + 1]:is(AST.Number) then
    --        ctxt:error(ast[2], "Compiler could not verify that "..
    --            "the matrix argument (2nd) to Affine is constant")
    --    end
    --end end

    return keyT(dst_rel)
end
local terra full_mod(val : int64, modulus : int64) : uint64
    return ((val % modulus) + modulus) % modulus
end
function B.Affine.codegen(ast, ctxt)
    local args      = ast.params
    local dst_rel   = args[1].node_type.value
    local src_rel   = args[3].node_type.relation
    local dst_dims  = dst_rel:Dims()
    local src_dims  = src_rel:Dims()
    local dst_wrap  = dst_rel:Periodicity()
    local nrow      = args[2].node_type.Nrow
    local ncol      = args[2].node_type.Ncol

    local srckey    = symbol(args[3].node_type:terratype())
    local mat       = symbol(args[2].node_type:terratype())
    local dsttype   = keyT(dst_rel):terratype()
    local results   = {}

    -- matrix multiply build
    for yi = 0,nrow-1 do
        -- read out the constant offset
        local sum = `mat.d[yi][ncol-1]
        for xi = 0,ncol-2 do
            local astr = 'a'..tostring(xi)
            sum = `[sum] + mat.d[yi][xi] * srckey.[astr]
        end
        -- make sure to clamp back down into address values
        sum = `[uint64](sum)
        -- add periodicity wrapping if specified
        if dst_wrap[yi+1] then
            results[yi+1] = `full_mod(sum, [ dst_dims[yi+1] ])
        else
            results[yi+1] = sum
        end
    end

    -- capture the arguments safely
    local wrapped = quote
        var [srckey] = [ args[3]:codegen(ctxt) ]
        var [mat]    = [ args[2]:codegen(ctxt) ]
    in
        [dsttype]({ results })
    end
    return wrapped
end

B.UNSAFE_ROW = Builtin.new()
function B.UNSAFE_ROW.check(ast, ctxt)
    local args = ast.params
    if #args ~= 2 then
        ctxt:error(ast, "UNSAFE_ROW expects exactly 2 arguments "..
                        "(instead got " .. #args .. ")")
        return errorT
    end

    local ret_type = nil

    local addr_type = args[1].node_type
    local rel_type = args[2].node_type
    if not rel_type:isinternal() or not is_relation(rel_type.value) then
        ctxt:error(ast, "UNSAFE_ROW expected a relation as the second arg")
        ret_type = errorT
    end
    local rel = rel_type.value
    local ndim = rel:nDims()
    --if rel:isGrid() then
    --    ctxt:error(ast, "UNSAFE_ROW cannot generate keys into a grid")
    --    ret_type = errorT
    --end
    if ndim == 1 and addr_type ~= uint64T then
        ctxt:error(ast, "UNSAFE_ROW expected a uint64 as the first arg")
        ret_type = errorT
    elseif ndim > 1  and addr_type ~= vectorT(uint64T,ndim) then
        ctxt:error(ast, "UNSAFE_ROW expected a vector of "..ndim..
                        " uint64 values")
    end

    -- actual typing
    if not ret_type then
        ret_type = keyT(rel_type.value)
    end
    return ret_type
end
function B.UNSAFE_ROW.codegen(ast, ctxt)
    local rel = ast.params[2].node_type.value
    local ndim = rel:nDims()
    local addrtype = keyT(rel):terratype()
    if ndim == 1 then
        return `[addrtype]({ [ast.params[1]:codegen(ctxt)] })
    else
        local vecui = ast.params[1]:codegen(ctxt)
        if ndim == 2 then
            return quote var v = vecui in
                [addrtype]({ v.d[0], v.d[1] })
            end
        else
            return quote var v = vecui in
                [addrtype]({ v.d[0], v.d[1], v.d[2] })
            end
        end
    end
end


B.assert = Builtin.new(assert)
function B.assert.check(ast, ctxt)
    local args = ast.params
    if #args ~= 1 then
        ctxt:error(ast, "assert expects exactly 1 argument (instead got " .. #args .. ")")
        return
    end

    local test = args[1]
    local test_type = test.node_type
    if test_type:isvector() then test_type = test_type:basetype() end
    if test_type ~= errorT and test_type ~= boolT then
        ctxt:error(ast, "expected a boolean or vector of booleans as the test for assert statement")
    end
end

local terra ebbAssert(test : bool, file : rawstring, line : int)
    if not test then
        C.fprintf(C.stderr, "%s:%d: assertion failed!\n", file, line)
        C.exit(1)
    end
end

-- NADA FOR NOW
local gpuAssert = terra(test : bool, file : rawstring, line : int) end

if terralib.cudacompile then
    gpuAssert = terra(test : bool, file : rawstring, line : int)
        if not test then
            G.printf("%s:%d: assertion failed!\n", file, line)
            cudalib.nvvm_membar_gl()
            terralib.asm(terralib.types.unit,"trap;","",true)
            --@([&uint8](0)) = 0 -- Should replace with a CUDA trap..../li
        end
    end
end

function B.assert.codegen(ast, ctxt)
    local test = ast.params[1]
    local code = test:codegen(ctxt)
    
    local tassert = ebbAssert
    if ctxt:onGPU() then tassert = gpuAssert end

    if test.node_type:isvector() then
        local N      = test.node_type.N
        local vec    = symbol(test.node_type:terratype())

        local all    = `true
        for i = 0, N-1 do
            all = `all and vec.d[i]
        end

        return quote
            var [vec] = [code]
        in
            tassert(all, ast.filename, ast.linenumber)
        end
    else
        return quote tassert(code, ast.filename, ast.linenumber) end
    end
end

B.print = Builtin.new(print)
function B.print.check(ast, ctxt)
    local args = ast.params
    
    for i,output in ipairs(args) do
        local outtype = output.node_type
        if outtype ~= errorT and
           not outtype:isvalue() and not outtype:iskey()
        then
            ctxt:error(ast, "only numbers, bools, vectors, matrices and keys can be printed")
        end
    end
end

local function printSingle (bt, exp, elemQuotes)
    if bt == floatT or bt == doubleT then
        table.insert(elemQuotes, `[double]([exp]))
        return "%f"
    elseif bt == intT then
        table.insert(elemQuotes, exp)
        return "%d"
    elseif bt == uint64T then 
        table.insert(elemQuotes, exp)
        return "%lu"
    elseif bt == boolT then
        table.insert(elemQuotes, `terralib.select([exp], "true", "false"))
        return "%s"
    else
        error('Unrecognized type in print: ' .. bt:toString() .. ' ' .. tostring(bt:terratype()))
    end
end

local function buildPrintSpec(ctxt, output, printSpec, elemQuotes, definitions)
    local lt   = output.node_type
    local tt   = lt:terratype()
    local code = output:codegen(ctxt)

    if lt:isvector() then
        printSpec = printSpec .. "{"
        local sym = symbol()
        local bt = lt:basetype()
        definitions = quote
            [definitions]
            var [sym] : tt = [code]
        end
        for i = 0, lt.N - 1 do
            printSpec = printSpec .. ' ' ..
                        printSingle(bt, `sym.d[i], elemQuotes)
        end
        printSpec = printSpec .. " }"

    elseif lt:ismatrix() then
        printSpec = printSpec .. '{'
        local sym = symbol()
        local bt = lt:basetype()
        definitions = quote
            [definitions]
            var [sym] : tt = [code]
        end
        for i = 0, lt.Nrow - 1 do
            printSpec = printSpec .. ' {'
            for j = 0, lt.Ncol - 1 do
                printSpec = printSpec .. ' ' ..
                            printSingle(bt, `sym.d[i][j], elemQuotes)
            end
            printSpec = printSpec .. ' }'
        end
        printSpec = printSpec .. ' }'
    elseif lt:isscalarkey() then
        if lt.ndims == 1 then
            printSpec = printSpec ..
                        printSingle(uint64T, `code.a0, elemQuotes)
        else
            local sym = symbol(lt:terratype())
            definitions = quote
                [definitions]
                var [sym] = [code]
            end
            printSpec = printSpec .. '{'
            for i = 0, lt.ndims-1 do
                local astr = 'a'..tostring(i)
                printSpec = printSpec .. ' ' ..
                            printSingle(uint64T, `sym.[astr], elemQuotes)
            end
            printSpec = printSpec .. ' }'
        end
    elseif lt:isvalue() then
        printSpec = printSpec .. printSingle(lt, code, elemQuotes)
    else
        assert(false and "printed object should always be number, bool, or vector")
    end
    return printSpec, elemQuotes, definitions
end

function B.print.codegen(ast, ctxt)
    local printSpec   = ''
    local elemQuotes  = {}
    local definitions = quote end

    for i,output in ipairs(ast.params) do
        printSpec, elemQuotes, definitions = buildPrintSpec(ctxt, output, printSpec, elemQuotes, definitions)
        local t = ast.params[i+1] and " " or ""
        printSpec = printSpec .. t
    end
    printSpec = printSpec .. "\n"

    local printf = C.printf
    if (ctxt:onGPU()) then
        printf = G.printf
    end
	return quote [definitions] in printf(printSpec, elemQuotes) end
end



B.rand = Builtin.new()
function B.rand.check(ast, ctxt)
    local args = ast.params
    if #args ~= 0 then
        ctxt:error(ast, "rand expects 0 arguments")
        return errorT
    else
        return doubleT
    end
end
local cpu_randinitialized = false
function B.rand.codegen(ast, ctxt)
    local rand
    local RAND_MAX
    if ctxt:onGPU() then
        -- GPU_util makes sure we're seeded for us
        rand = `G.rand([ctxt:gid()])
        RAND_MAX = G.RAND_MAX
    else
        -- make sure we're seeded
        if not cpu_randinitialized then
            cpu_randinitialized = true
            C.srand(0) --cmath.time(nil)
        end
        rand = `C.rand()
        RAND_MAX = C.RAND_MAX
    end

    -- reciprocal
    --local reciprocal = terralib.constant(double, 1/(1.0e32-1.0))
    return `[double](rand) / RAND_MAX
end
-- SHOULD expose some randomness controls here...



B.dot = Builtin.new()
function B.dot.check(ast, ctxt)
    local args = ast.params
    if #args ~= 2 then
        ctxt:error(ast, "dot product expects exactly 2 arguments "..
                        "(instead got " .. #args .. ")")
        return errorT
    end

    local lt1 = args[1].node_type
    local lt2 = args[2].node_type

    local numvec_err = 'arguments to dot product must be numeric vectors'
    local veclen_err = 'vectors in dot product must have equal dimensions'
    if not lt1:isvector()  or not lt2:isvector() or
       not lt1:isnumeric() or not lt2:isnumeric()
    then
        ctxt:error(ast, numvec_err)
    elseif lt1.N ~= lt2.N then
        ctxt:error(ast, veclen_err)
    else
        return T.type_join(lt1:basetype(), lt2:basetype())
    end

    return errorT
end

function B.dot.codegen(ast, ctxt)
    local args = ast.params

    local N     = args[1].node_type.N
    local lhtyp = args[1].node_type:terratype()
    local rhtyp = args[2].node_type:terratype()
    local lhe   = args[1]:codegen(ctxt)
    local rhe   = args[2]:codegen(ctxt)

    local lhval = symbol(lhtyp)
    local rhval = symbol(rhtyp)

    local exp = `0
    for i = 0, N-1 do
        exp = `exp + lhval.d[i] * rhval.d[i]
    end

    return quote
        var [lhval] = [lhe]
        var [rhval] = [rhe]
    in
        exp
    end
end



B.cross = Builtin.new()
function B.cross.check(ast, ctxt)
    local args = ast.params

    if #args ~= 2 then
        ctxt:error(ast, "cross product expects exactly 2 arguments "..
                        "(instead got " .. #args .. ")")
        return errorT
    end

    local lt1 = args[1].node_type
    local lt2 = args[2].node_type

    local numvec_err = 'arguments to cross product must be numeric vectors'
    local veclen_err = 'vectors in cross product must be 3 dimensional'
    if not lt1:isvector()  or not lt2:isvector() or
       not lt1:isnumeric() or not lt2:isnumeric()
    then
        ctxt:error(ast, numvec_err)
    elseif lt1.N ~= 3 or lt2.N ~= 3 then
        ctxt:error(ast, veclen_err)
    else
        return T.type_join(lt1, lt2)
    end

    return errorT
end

function B.cross.codegen(ast, ctxt)
    local args = ast.params

    local lhtyp = args[1].node_type:terratype()
    local rhtyp = args[2].node_type:terratype()
    local lhe   = args[1]:codegen(ctxt)
    local rhe   = args[2]:codegen(ctxt)

    local typ = T.type_join(args[1].node_type, args[2].node_type)

    return quote
        var lhval : lhtyp = [lhe]
        var rhval : rhtyp = [rhe]
    in 
        [typ:terratype()]({ arrayof( [typ:terrabasetype()],
            lhval.d[1] * rhval.d[2] - lhval.d[2] * rhval.d[1],
            lhval.d[2] * rhval.d[0] - lhval.d[0] * rhval.d[2],
            lhval.d[0] * rhval.d[1] - lhval.d[1] * rhval.d[0]
        )})
    end
end



B.length = Builtin.new()
function B.length.check(ast, ctxt)
    local args = ast.params
    if #args ~= 1 then
        ctxt:error(ast, "length expects exactly 1 argument (instead got " .. #args .. ")")
        return errorT
    end
    local lt = args[1].node_type
    if not lt:isvector() then
        ctxt:error(args[1], "argument to length must be a vector")
        return errorT
    end
    if not lt:basetype():isnumeric() then
        ctxt:error(args[1], "length expects vectors of numeric type")
    end
    if lt:basetype() == floatT then return floatT
                                else return doubleT end
end

function B.length.codegen(ast, ctxt)
    local args = ast.params

    local N      = args[1].node_type.N
    local typ    = args[1].node_type:terratype()
    local exp    = args[1]:codegen(ctxt)

    local vec    = symbol(typ)

    local len2 = `0
    for i = 0, N-1 do
        len2 = `len2 + vec.d[i] * vec.d[i]
    end

    local sqrt = C.sqrt
    if ctxt:onGPU() then sqrt = G.sqrt end

    return quote
        var [vec] = [exp]
    in
        sqrt( len2 )
    end
end

function Builtin.newDoubleFunction(name)
    local cpu_fn = C[name]
    local gpu_fn = G[name]
    local lua_fn = function (arg) return cpu_fn(arg) end

    local b = Builtin.new(lua_fn)

    function b.check (ast, ctxt)
        local args = ast.params
        if #args ~= 1 then
            ctxt:error(ast, name.." expects exactly 1 argument "..
                            "(instead got ".. #args ..")")
            return errorT
        end
        local lt = args[1].node_type
        if not lt:isnumeric() then
            ctxt:error(args[1], "argument to "..name.." must be numeric")
        end
        if lt:isvector() then
            ctxt:error(args[1], "argument to "..name.." must be a scalar")
            return errorT
        end
        return doubleT
    end

    function b.codegen (ast, ctxt)
        local exp = ast.params[1]:codegen(ctxt)
        if ctxt:onGPU() then
            return `gpu_fn([exp])
        else
            return `cpu_fn([exp])
        end
    end
    return b
end

L.cos   = Builtin.newDoubleFunction('cos')
L.acos  = Builtin.newDoubleFunction('acos')
L.sin   = Builtin.newDoubleFunction('sin')
L.asin  = Builtin.newDoubleFunction('asin')
L.tan   = Builtin.newDoubleFunction('tan')
L.atan  = Builtin.newDoubleFunction('atan')
L.sqrt  = Builtin.newDoubleFunction('sqrt')
L.cbrt  = Builtin.newDoubleFunction('cbrt')
L.floor = Builtin.newDoubleFunction('floor')
L.ceil  = Builtin.newDoubleFunction('ceil')
L.fabs  = Builtin.newDoubleFunction('fabs')
L.log   = Builtin.newDoubleFunction('log')

L.fmin  = Builtin.new()
L.fmax  = Builtin.new()
local function minmax_check(ast, ctxt, name)
    if #ast.params ~= 2 then
        ctxt:error(ast, name.." expects 2 arguments "..
                        "(instead got ".. #ast.params ..")")
        return errorT
    end
    local lt = ast.params[1].node_type
    local rt = ast.params[1].node_type
    if not lt:isnumeric() or not rt:isscalar() then
        ctxt:error(ast.params[1], "argument to "..name..
                                  " must be a scalar number")
    end
    if not rt:isnumeric() or not rt:isscalar() then
        ctxt:error(ast.params[2], "argument to "..name..
                                  " must be a scalar number")
    end
    return doubleT
end
function L.fmin.check(ast, ctxt) return minmax_check(ast, ctxt, 'fmin') end
function L.fmax.check(ast, ctxt) return minmax_check(ast, ctxt, 'fmax') end
function L.fmin.codegen(ast, ctxt)
    local lhs, rhs = ast.params[1]:codegen(ctxt), ast.params[2]:codegen(ctxt)
    if ctxt:onGPU() then return `G.fmin(lhs, rhs)
                    else return `C.fmin(lhs, rhs) end
end
function L.fmax.codegen(ast, ctxt)
    local lhs, rhs = ast.params[1]:codegen(ctxt), ast.params[2]:codegen(ctxt)
    if ctxt:onGPU() then return `G.fmax(lhs, rhs)
                    else return `C.fmax(lhs, rhs) end
end


--terra b_and (a : int, b : int)
--    return a and b
--end
--L.band = Builtin.new(b_and)
--function L.band.check (ast, ctxt)
--    local args = ast.params
--    if #args ~= 2 then ctxt:error(ast, "binary_and expects 2 arguments "..
--                                       "(instead got " .. #args .. ")")
--        return errorT
--    end
--    for i = 1, #args do
--        local lt = args[i].node_type
--        if lt ~= intT then
--            ctxt:error(args[i], "argument "..i..
--                                "to binary_and must be numeric")
--            return errorT
--        end
--    end
--    return intT
--end
--function L.band.codegen(ast, ctxt)
--    local exp1 = ast.params[1]:codegen(ctxt)
--    local exp2 = ast.params[2]:codegen(ctxt)
--    return `b_and([exp1], [exp2])
--end

L.pow = Builtin.new(C.pow)
function L.pow.check (ast, ctxt)
    local args = ast.params
    if #args ~= 2 then ctxt:error(ast, "pow expects 2 arguments (instead got " .. #args .. ")")
        return errorT
    end
    for i = 1, #args do
        local lt = args[i].node_type
        if not lt:isnumeric() then
            ctxt:error(args[i], "argument "..i.." to pow must be numeric")
            return errorT
        end
    end
    for i = 1, #args do
        local lt = args[i].node_type
        if not lt:isscalar() then
            ctxt:error(args[i], "argument "..i.." to pow must be a scalar")
            return errorT
        end
    end
    return doubleT
end
function L.pow.codegen(ast, ctxt)
    local exp1 = ast.params[1]:codegen(ctxt)
    local exp2 = ast.params[2]:codegen(ctxt)
    if ctxt:onGPU() then
        return `G.pow([exp1], [exp2])
    else
        return `C.pow([exp1], [exp2])
    end
end

L.fmod = Builtin.new(C.fmod)
function L.fmod.check (ast, ctxt)
    local args = ast.params
    if #args ~= 2 then ctxt:error(ast, "fmod expects 2 arguments (instead got " .. #args .. ")")
        return errorT
    end
    for i = 1, #args do
        local lt = args[i].node_type
        if not lt:isnumeric() then
            ctxt:error(args[i], "argument "..i.." to fmod must be numeric")
            return errorT
        end
    end
    for i = 1, #args do
        local lt = args[i].node_type
        if not lt:isscalar() then
            ctxt:error(args[i], "argument "..i.." to fmod must be a scalar")
            return errorT
        end
    end
    return doubleT
end
function L.fmod.codegen(ast, ctxt)
    local exp1 = ast.params[1]:codegen(ctxt)
    local exp2 = ast.params[2]:codegen(ctxt)
    if ctxt:onGPU() then
        return `G.fmod([exp1], [exp2])
    else
        return `C.fmod([exp1], [exp2])
    end
end



B.all = Builtin.new()
function B.all.check(ast, ctxt)
    local args = ast.params
    if #args ~= 1 then
        ctxt:error(ast, "all expects exactly 1 argument (instead got " .. #args .. ")")
        return errorT
    end
    local lt = args[1].node_type
    if not lt:isvector() then
        ctxt:error(args[1], "argument to all must be a vector")
        return errorT
    end
    return boolT
end

function B.all.codegen(ast, ctxt)
    local args = ast.params

    local N      = args[1].node_type.N
    local typ    = args[1].node_type:terratype()
    local exp    = args[1]:codegen(ctxt)

    local val    = symbol(typ)

    local outexp = `true
    for i = 0, N-1 do
        outexp = `outexp and val.d[i]
    end

    return quote
        var [val] = [exp]
    in
        outexp
    end
end



B.any = Builtin.new()
function B.any.check(ast, ctxt)
    local args = ast.params
    if #args ~= 1 then
        ctxt:error(ast, "any expects exactly 1 argument (instead got " .. #args .. ")")
        return errorT
    end
    local lt = args[1].node_type
    if not lt:isvector() then
        ctxt:error(args[1], "argument to any must be a vector")
        return errorT
    end
    return boolT
end

function B.any.codegen(ast, ctxt)
    local args   = ast.params

    local N      = args[1].node_type.N
    local typ    = args[1].node_type:terratype()
    local exp    = args[1]:codegen(ctxt)

    local val    = symbol(typ)

    local outexp = `false
    for i = 0, N-1 do
        outexp = `outexp or val.d[i]
    end

    return quote
        var [val] = [exp]
    in
        outexp
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
    
    return symbol(arg.node_type:terratype())
end

local function TerraCheck(func)
    return function (ast, ctxt)
        local args = ast.params
        local argsyms = map(GetTypedSymbol, args)
        local rettype = nil
        local try = function()
            local terrafunc = terra([argsyms]) return func([argsyms]) end
            rettype = terrafunc:getdefinitions()[1]:gettype().returntype
        end
        local success, retval = pcall(try)
        if not success then
            ctxt:error(ast, "couldn't fit parameters to signature of terra function")
            ctxt:error(ast, retval)
            return errorT
        end
        -- Kinda terrible hack due to flux in Terra inteface here
        if rettype:isstruct() and terralib.sizeof(rettype) == 0 then
            -- case of no return value, no type is needed
            return
        end
        if not T.terraToEbbType(rettype) then
            ctxt:error(ast, "unable to use return type '"..tostring(rettype)..
                            "' of terra function in Ebb")
            return errorT
        end
        return T.terraToEbbType(rettype)
    end
end

local function TerraCodegen(func)
    return function (ast, ctxt)
        local args = ast.params
        local argsyms = map(GetTypedSymbol, args)
        local init_params = quote end
        for i = 1, #args do
            local code = args[i]:codegen(ctxt)
            init_params = quote
                init_params
                var [argsyms[i]] = code
            end
        end
        return quote init_params in func(argsyms) end
    end 
end


function B.terra_to_func(terrafn)
    local newfunc = Builtin.new()
    newfunc.is_a_terra_func = true
    newfunc.check, newfunc.codegen = TerraCheck(terrafn), TerraCodegen(terrafn)
    return newfunc
end

