local GPU = {}
package.loaded['compiler.gpu_util'] = GPU

if not terralib.cudacompile then return end

local tid_x   = cudalib.nvvm_read_ptx_sreg_tid_x
local tid_y   = cudalib.nvvm_read_ptx_sreg_tid_y
local tid_z   = cudalib.nvvm_read_ptx_sreg_tid_z

local b_dim_x = cudalib.nvvm_read_ptx_sreg_ntid_x
local b_dim_y = cudalib.nvvm_read_ptx_sreg_ntid_y
local b_dim_z = cudalib.nvvm_read_ptx_sreg_ntid_z

local bid_x   = cudalib.nvvm_read_ptx_sreg_ctaid_x
local bid_y   = cudalib.nvvm_read_ptx_sreg_ctaid_y
local bid_z   = cudalib.nvvm_read_ptx_sreg_ctaid_z

local g_dim_x = cudalib.nvvm_read_ptx_sreg_nctaid_x
local g_dim_y = cudalib.nvvm_read_ptx_sreg_nctaid_y
local g_dim_z = cudalib.nvvm_read_ptx_sreg_nctaid_z

local thread_id = macro(function()
    return `(tid_x() +
             tid_y() * b_dim_x() +
             tid_z() * b_dim_x() * b_dim_y())
end)

local block_id = macro(function()
	return `(bid_x() +
             bid_y() * g_dim_x() + 
             bid_z() * g_dim_x() * g_dim_y())
end)

local num_blocks = macro(function()
	return `(g_dim_x()*g_dim_y()*g_dim_z())
end)

local global_tid = macro(function()
    return `(thread_id()+block_id()*num_blocks())
end)

local terra get_grid_dimensions (num_blocks : uint64, max_grid_dim : uint64) : {uint, uint, uint}
    if num_blocks < max_grid_dim then
        return { num_blocks, 1, 1 }
    elseif num_blocks / max_grid_dim < max_grid_dim then
        return { max_grid_dim, (num_blocks + max_grid_dim - 1) / max_grid_dim, 1 }
    else
        return { max_grid_dim, max_grid_dim, (num_blocks - 1) / max_grid_dim / max_grid_dim + 1 }
    end
end

local vprintf = terralib.externfunction("cudart:vprintf", {&int8,&int8} -> int)
local function createbuffer(args)
    local Buf = terralib.types.newstruct()
    return quote
        var buf : Buf
        escape
            for i,e in ipairs(args) do
                local typ = e:gettype()
                local field = "_"..tonumber(i)
                typ = typ == float and double or typ
                table.insert(Buf.entries,{field,typ})
                emit quote
                   buf.[field] = e
                end
            end
        end
    in
        [&int8](&buf)
    end
end

local printf = macro(function(fmt,...)
    local buf = createbuffer({...})
    return `vprintf(fmt,buf) 
end)

local reduce_max_int32 = macro(function(result_ptr, val)
    return terralib.asm(terralib.types.unit,"red.global.max.u32 [$0], $1;","l,r",true,result_ptr,val)
end)

local reduce_min_int32 = macro(function(result_ptr, val)
    return terralib.asm(terralib.types.unit,"red.global.min.u32 [$0], $1;","l,r",true,result_ptr,val)
end)

local reduce_add_int32 = macro(function(result_ptr, val)
    return terralib.asm(terralib.types.unit,"red.global.add.u32 [$0], $1;","l,r",true,result_ptr,val)
end)

local reduce_and_b32 = macro(function(result_ptr, val)
    return terralib.asm(terralib.types.unit,"red.global.and.b32 [$0], $1;","l,r",true,result_ptr,val)
end)

local reduce_or_b32 = macro(function(result_ptr, val)
    return terralib.asm(terralib.types.unit,"red.global.or.b32 [$0], $1;","l,r",true,result_ptr,val)
end)

--[[------------------------------------------------------------------------]]--
--[[ Implementation of slow atomics                                         ]]--
--[[------------------------------------------------------------------------]]--
local cas_uint64 = terra(address : &uint64, compare : uint64, value : uint64)
    var old : uint64 = @address
    terralib.asm(terralib.types.uint64, "atom.global.cas.b64 $0, [$1], $2, $3;","l,l,l,l",true,old,address,compare,value)
    return old
end

local cas_uint32 = terra(address : &uint32, compare : uint32, value : uint32)
    var old : uint32 = @address
    terralib.asm(terralib.types.uint64, "atom.global.cas.b64 $0, [$1], $2, $3;","r,l,r,r",true,old,address,compare,value)
    return old
end

local function generate_slow_atomic_64 (op, typ)
    return terra (address : &typ, operand : typ)
        var old : typ = @address
        var assumed : typ
        var mask = false
        repeat
            if not mask then
                assumed = old
                old     = cas_uint64(address, assumed, op(assumed,operand))
                mask    = assumed == old
            end
        until mask
    end
end

local function generate_slow_atomic_32 (op, typ)
    return terra (address : &typ, operand : typ)
        var old : typ = @address
        var assumed : typ
        var mask = false
        repeat
            if not mask then
                assumed = old
                old     = cas_uint32(address, assumed, op(assumed,operand))
                mask    = assumed == old
            end
        until mask
    end
end

local mul = macro(function(a, b) return `a*b end)
local add = macro(function(a, b) return `a+b end)
local max = macro(function(a, b) return
    quote 
        var max
        if a > b then max = a
        else          max = b
        end
    in
        max
    end
end)
local min = macro(function(a, b) return
    quote 
        var min
        if a < b then min = a
        else          min = b
        end
    in
        min
    end
end)

local atomic_mult_double = terra (address : &double, factor : double)
    var old : double = @address
    var assumed : double
    var mask = false
    repeat
        if not mask then
            assumed = old
            old     = cas_uint64(address, assumed, assumed*factor)
            mask = assumed == old
        end
    until mask
end

local atomic_mult_uint64 = terra(address : &uint64, factor : uint64)
    var old : uint64 = @address
    var assumed : uint64
    var mask = false
    repeat
        if not mask then
            assumed = old
            old     = cas_uint64(address, assumed, assumed*factor)
            mask = assumed == old
        end
    until mask
end

GPU.printf     = printf
GPU.block_id   = block_id
GPU.thread_id  = thread_id
GPU.global_tid = global_tid
GPU.num_blocks = num_blocks

GPU.barrier    = macro(function() return quote cudalib.ptx_bar_sync(0) end end)
GPU.sync       = terralib.externfunction("cudaThreadSynchronize", {} -> int)

GPU.get_grid_dimensions = get_grid_dimensions

-- Intrinsic atomic reductions
GPU.atomic_add_float = terralib.intrinsic("llvm.nvvm.atomic.load.add.f32.p0f32", {&float,float} -> {float})
GPU.reduce_max_int32 = reduce_max_int32
GPU.reduce_min_int32 = reduce_min_int32
GPU.reduce_add_int32 = reduce_add_int32
GPU.reduce_and_b32   = reduce_and_b32
GPU.reduce_or_b32    = reduce_or_b32

-- Slow operations!
GPU.atomic_mult_float_SLOW  = generate_slow_atomic_32(mul,float)
GPU.atomic_mult_double_SLOW = generate_slow_atomic_64(mul,double)

GPU.atomic_mult_int32_SLOW  = generate_slow_atomic_32(mul,int32)
GPU.atomic_mult_uint64_SLOW = generate_slow_atomic_64(mul,uint64)

GPU.atomic_add_uint64_SLOW = generate_slow_atomic_64(add,uint64)
GPU.atomic_add_double_SLOW = generate_slow_atomic_64(add,double)

GPU.atomic_min_uint64_SLOW = generate_slow_atomic_64(min,uint64)
GPU.atomic_min_double_SLOW = generate_slow_atomic_64(min,double)
GPU.atomic_min_float_SLOW  = generate_slow_atomic_32(min,float)

GPU.atomic_max_uint64_SLOW = generate_slow_atomic_64(max,uint64)
GPU.atomic_max_double_SLOW = generate_slow_atomic_64(max,double)
GPU.atomic_max_float_SLOW  = generate_slow_atomic_32(max,float)
