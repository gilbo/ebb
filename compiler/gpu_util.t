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

local block_id = macro(function()
	return `(bid_x() +
             bid_y() * g_dim_x() + 
             bid_z() * g_dim_x() * g_dim_y())
end)

local thread_id = macro(function()
	return `(tid_x() +
		     tid_y() * b_dim_x() +
		     tid_z() * b_dim_x() * b_dim_y())
end)

local num_blocks = macro(function()
	return `(g_dim_x()*g_dim_y()*g_dim_z())
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


GPU.printf     = printf
GPU.block_id   = block_id
GPU.thread_id  = thread_id
GPU.num_blocks = num_blocks

GPU.get_grid_dimensions = get_grid_dimensions
