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

GPU.printf     = printf
GPU.block_id   = block_id
GPU.thread_id  = thread_id
GPU.global_tid = global_tid
GPU.num_blocks = num_blocks

GPU.barrier    = macro(function() return quote cudalib.ptx_bar_sync(0) end end)
GPU.sync       = terralib.externfunction("cudaThreadSynchronize", {} -> int)

GPU.get_grid_dimensions = get_grid_dimensions

GPU.atomic_add_float = terralib.intrinsic("llvm.nvvm.atomic.load.add.f32.p0f32", {&float,float} -> {float})
GPU.reduce_max_int32 = reduce_max_int32
GPU.reduce_min_int32 = reduce_min_int32
GPU.reduce_add_int32 = reduce_add_int32
GPU.reduce_and_b32 = reduce_and_b32
GPU.reduce_or_b32  = reduce_or_b32
