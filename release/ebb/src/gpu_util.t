local GPU = {}
package.loaded['ebb.src.gpu_util'] = GPU
local C   = require 'ebb.src.c'

if not terralib.cudacompile then return end

local WARPSIZE = 32

--[[-----------------------------------------------------------------------]]--
--[[ Thread/Grid/Block                                                     ]]--
--[[-----------------------------------------------------------------------]]--
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

local terra get_grid_dimensions(
  num_blocks : uint64,
  max_grid_dim : uint64
) : {uint, uint, uint}
  if num_blocks < max_grid_dim then
    return { num_blocks, 1, 1 }
  elseif num_blocks / max_grid_dim < max_grid_dim then
    return { max_grid_dim,
             (num_blocks + max_grid_dim - 1) / max_grid_dim,
             1 }
  else
    return { max_grid_dim,
             max_grid_dim,
             (num_blocks - 1) / max_grid_dim / max_grid_dim + 1 }
  end
end


--[[-----------------------------------------------------------------------]]--
--[[ Print                                                                 ]]--
--[[-----------------------------------------------------------------------]]--
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


--[[-----------------------------------------------------------------------]]--
--[[ Math                                                                  ]]--
--[[-----------------------------------------------------------------------]]--
-- link the bitcode for libdevice so that we can access device math functions
-- CUDA libdevice has all the math functions:
-- http://docs.nvidia.com/cuda/libdevice-users-guide/#axzz3CND85k3B
local cuda_success, cuda_version =
  pcall(function() return cudalib.localversion() end)
cuda_version = cuda_success and cuda_version or 30

local externcall = terralib.externfunction
local libdevice = terralib.cudahome..
  string.format("/nvvm/libdevice/libdevice.compute_%d.10.bc",cuda_version)
if terralib.linkllvm then
  local llvmbc = terralib.linkllvm(libdevice)
  externcall = function(name, ftype)
    return llvmbc:extern(name, ftype)
  end
else
  terralib.linklibrary(libdevice)
end

local cbrt  = externcall("__nv_cbrt", double -> double)
local sqrt  = externcall("__nv_sqrt", double -> double)
local cos   = externcall("__nv_cos",  double -> double)
local acos  = externcall("__nv_acos", double -> double)
local sin   = externcall("__nv_sin",  double -> double)
local asin  = externcall("__nv_asin", double -> double)
local tan   = externcall("__nv_tan",  double -> double)
local atan  = externcall("__nv_atan", double -> double)
local log   = externcall("__nv_log",  double -> double)
local pow   = externcall("__nv_pow",  {double, double} -> double)
local fmod  = externcall("__nv_fmod", {double, double} -> double)
local floor = externcall("__nv_floor", double -> double)
local ceil  = externcall("__nv_ceil", double -> double)
local fabs  = externcall("__nv_fabs", double -> double)

local fmin  = externcall("__nv_fmin", {double,double} -> double)
local fmax  = externcall("__nv_fmax", {double,double} -> double)


local terra popc_b32(bits : uint32) : uint32
  return terralib.asm(terralib.types.uint32,
    "popc.b32  $0, $1;","=r,r",false,bits)
end

local terra brev_b32(bits : uint32) : uint32
  return terralib.asm(terralib.types.uint32,
    "brev.b32  $0, $1;","=r,r",false,bits)
end
local terra clz_b32(bits : uint32) : uint32
  return terralib.asm(terralib.types.uint32,
    "clz.b32  $0, $1;","=r,r",false,bits)
end

--[[-----------------------------------------------------------------------]]--
--[[ Atomic reductions                                                     ]]--
--[[-----------------------------------------------------------------------]]--
local terra reduce_max_int32(address : &int32, operand : int32)
  terralib.asm(terralib.types.unit,
    "red.global.max.s32 [$0], $1;","l,r",true,address,operand)
end

local terra reduce_min_int32(address : &int32, operand : int32)
  terralib.asm(terralib.types.unit,
    "red.global.min.s32 [$0], $1;","l,r",true,address,operand)
end

local terra reduce_add_int32(address : &int32, operand : int32)
  terralib.asm(terralib.types.unit,
    "red.global.add.s32 [$0], $1;","l,r",true,address,operand)
end

--[[
-- doubt this will work right
local terra reduce_and_b32(address : &bool, operand : bool)
  terralib.asm(terralib.types.unit,
    "red.global.and.b32 [$0], $1;","l,r",true,address,operand)
end)

-- doubt this will work right
local terra reduce_or_b32(address : &bool, operand : bool)
  terralib.asm(terralib.types.unit,
    "red.global.or.b32 [$0], $1;","l,r",true,address,operand)
end)
--]]

-- presumably this should work too?
local terra reduce_max_f32(address : &float, operand : float)
  terralib.asm(terralib.types.unit,
    "red.global.max.f32 [$0], $1;","l,f",true,address,operand)
end

-- presumably this should work too?
local terra reduce_min_f32(address : &float, operand : float)
  terralib.asm(terralib.types.unit,
    "red.global.min.f32 [$0], $1;","l,f",true,address,operand)
end

local atomic_add_float =
  terralib.intrinsic("llvm.nvvm.atomic.load.add.f32.p0f32",
                     {&float,float} -> {float})

local terra atomic_add_uint64(address : &uint64, operand : uint64) : uint64
  return terralib.asm(terralib.types.uint64,
    "atom.global.add.u64 $0, [$1], $2;","=l,l,l",true,address,operand)
end
local terra reduce_add_uint64(address : &uint64, operand : uint64)
  terralib.asm(terralib.types.unit,
    "red.global.add.u64 [$0], $1;","l,l",true,address,operand)
end

--[[-----------------------------------------------------------------------]]--
--[[ Warp Level Instructions                                               ]]--
--[[-----------------------------------------------------------------------]]--

local terra warpballot_b32() : uint32
  return terralib.asm(terralib.types.uint32,
    "vote.ballot.b32 $0, 0xFFFFFFFF;","=r",false)
end

local terra shuffle_index_b32( input : uint32, idx : uint32 ) : uint32
  return terralib.asm(terralib.types.uint32,
    "shfl.idx.b32 $0, $1, $2, 0x1F;","=r,r,r",false,input,idx)
end

--[[-----------------------------------------------------------------------]]--
--[[ Implementation of slow atomics                                        ]]--
--[[-----------------------------------------------------------------------]]--
local terra cas_uint64(address : &uint64, compare : uint64, value : uint64)
  return terralib.asm(terralib.types.uint64,
    "atom.global.cas.b64 $0, [$1], $2, $3;",
    "=l,l,l,l",true,address,compare,value)
end

local terra cas_uint32(address : &uint32, compare : uint32, value : uint32)
  return terralib.asm(terralib.types.uint32,
    "atom.global.cas.32 $0, [$1], $2, $3;",
    "=r,l,r,r",true,address,compare,value)
end

local function generate_slow_atomic_64 (op, typ)
  return terra (address : &typ, operand : typ)
    var old : typ = @address
    var assumed : typ
    var new     : typ

    var new_b     : &uint64 = [&uint64](&new)
    var assumed_b : &uint64 = [&uint64](&assumed)
    var res       :  uint64

    var mask = false
    repeat
      if not mask then
        assumed = old
        new     = op(assumed,operand)
        res     = cas_uint64([&uint64](address), @assumed_b, @new_b)
        old     = @[&typ](&res)
        mask    = assumed == old
      end
    until mask
  end
end

local function generate_slow_atomic_32 (op, typ)
  return terra (address : &typ, operand : typ)
    var old : typ = @address
    var assumed   : typ
    var new       : typ

    var new_b     : &uint32 = [&uint32](&new)
    var assumed_b : &uint32 = [&uint32](&assumed)
    var res       :  uint32

    var mask = false
    repeat
      if not mask then
        assumed = old
        new     = op(assumed,operand)
        res     = cas_uint32([&uint32](address), @assumed_b, @new_b)
        old     = @[&typ](&res)
        mask    = assumed == old
      end
    until mask
  end
end

-- Operator quotes
local mul = macro(function(a, b) return `a*b end)
local add = macro(function(a, b) return `a+b end)
local div = macro(function(a, b) return `a/b end)
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

--[[-----------------------------------------------------------------------]]--
--[[ Convenience Functions                                                 ]]--
--[[-----------------------------------------------------------------------]]--
local function throw_err() error('Cuda error') end
local function cuda_checkpoint()
  print('CUDA CHECK HERE')
  print(debug.traceback())
end
local cuda_error_checking = macro(function(code)
  --local function say_code()
  --    code:printpretty()
  --end
  return quote
    --say_code()
    --cuda_checkpoint()
    if code ~= 0 then
      C.printf("CUDA ERROR: %s\n", C.cudaGetErrorString(code))
      throw_err()
    end
  end
end)

local terra cuda_terra_malloc(size : uint64)
  var r : &opaque
  cuda_error_checking(C.cudaMalloc(&r, size))
  return r
end

local terra cuda_terra_free(ptr : &opaque)
  cuda_error_checking(C.cudaFree(ptr))
end

local terra cuda_memcpy_cpu_from_gpu(dst:&opaque, src:&opaque, N:uint64)
  cuda_error_checking(C.cudaMemcpy(dst, src, N, C.cudaMemcpyDeviceToHost))
end
local terra cuda_memcpy_gpu_from_cpu(dst:&opaque, src:&opaque, N:uint64)
  cuda_error_checking(C.cudaMemcpy(dst, src, N, C.cudaMemcpyHostToDevice))
end
local terra cuda_memcpy_gpu_from_gpu(dst:&opaque, src:&opaque, N:uint64)
  cuda_error_checking(C.cudaMemcpy(dst, src, N, C.cudaMemcpyDeviceToDevice))
end

local terra cuda_peek_at_last_error()
  cuda_error_checking(C.cudaPeekAtLastError())
end

local sync_temp_wrapper =
        terralib.externfunction("cudaThreadSynchronize", {} -> int)
local terra cuda_sync_wrapper_with_peek()
  var res = sync_temp_wrapper()
  cuda_peek_at_last_error()
end



--[[-----------------------------------------------------------------------]]--
--[[ Global Reductions                                                     ]]--
--[[-----------------------------------------------------------------------]]--

local ReductionObj = {}
ReductionObj.__index = ReductionObj

function ReductionObj.New(args)
  local ro = setmetatable({
    _ttype              = args.ttype or assert(false,'no ttype'),
    _blocksize          = args.blocksize or assert(false,'no blocksize'),
    _reduce_ident       = args.reduce_ident
                          or assert(false,'no reduce_ident'),
    _reduce_binop       = args.reduce_binop
                          or assert(false,'no reduce_binop'),
    _gpu_reduce_atomic  = args.gpu_reduce_atomic
                          or assert(false,'no gpu_reduce_atomic'),
  }, ReductionObj)

  -- initialization of shared memory variable
  ro._sharedmem     = cudalib.sharedmemory(ro._ttype, ro._blocksize)
  ro._sharedmemsize = terralib.sizeof(ro._ttype) * ro._blocksize

  return ro
end

function ReductionObj:getSharedMemPtr()
  return self._sharedmem
end
function ReductionObj:sharedMemSize()
  return self._sharedmemsize
end
function ReductionObj:sharedMemInitCode(tid_sym)
  return quote
    [self._sharedmem][tid_sym] = [self._reduce_ident]
  end
end
-- returns a snippet of code to be included at the end of the kernel
function ReductionObj:sharedMemReductionCode(tid_sym, globalptr)
  -- Shared Memory Reduction Tree
  local code = quote escape
    local step = self._blocksize
    while step > 1 do
      step = step / 2
      emit quote
        if tid_sym < step then
          var exp = [self._reduce_binop(`[self._sharedmem][tid_sym],
                                        `[self._sharedmem][tid_sym + step])]
          terralib.attrstore(&[self._sharedmem][tid_sym],
                             exp, {isvolatile=true})
        end
        GPU.barrier()
      end
    end
  end end

  code = quote
    [code]
    if tid_sym == 0 then
      [ self._gpu_reduce_atomic( `@globalptr, `[self._sharedmem][0] ) ]
    end
  end
  return code
end


--[[-----------------------------------------------------------------------]]--
--[[ Write Buffer                                                          ]]--
--[[-----------------------------------------------------------------------]]--

-- Reserves a unique index using the @writeidxptr counter;
-- increments @writeidxptr atomically; runs at the warp level
local function reserve_idx(tidsym, writeidxptr)
  local code = quote
  -- First, we're going to number all of the threads writing uniquely.
  -- And get a total count
    var ballot      = warpballot_b32()
      -- total # of writes
    var write_count = popc_b32(ballot)
      -- generate a pattern of all 1s up to but excluding this thread
    var ones        = ([uint64](1) << (tidsym % 32)) - 1
      -- # of writes by threads with a lower #; densely orders active threads
    var active_num  = popc_b32(ballot and ones)
  -- Second, now that we can designate a thread as leader, reserve some space
    -- Lowest # active thread becomes the leader
    var leader_num  = clz_b32(brev_b32(ballot))
    var is_leader   = active_num == 0

    var start_idx : uint64 = 0
    if is_leader then
      -- leader reserves buffer space for the whole warp
      start_idx = atomic_add_uint64(writeidxptr, write_count)
    end
  --  -- need to scatter the start_byte value to all active threads in the warp
    start_idx = shuffle_index_b32(start_idx, leader_num)
  ---- Third, return this resulting value about where to write
  in
    start_idx + active_num
  end
  return code
end


--[[-----------------------------------------------------------------------]]--
--[[ Rand                                                                  ]]--
--[[-----------------------------------------------------------------------]]--
-- Taken from: http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
local size_randbuffer = 1e6 -- one million; ~ 4MB storage...
local randbuffer = terralib.cast(&uint32,
        cuda_terra_malloc( size_randbuffer*terralib.sizeof(uint32) ))
local currentseed = 0
local randisinitialized = false
local RAND_MAX = 1.0e32-1.0

local function seedrand( globalseed )
  randisinitialized = true
  currentseed = globalseed

  local terra rand_init_body()
    var tid = [uint32](global_tid())
    if tid < size_randbuffer then
      -- compute a seed from the thread id and the global seed value
      var seed = tid + [uint32](globalseed)
      -- do the wang hash to get more uncorrelated initial seed values
      seed = (seed ^ 61) ^ (seed >> 16)
      seed = seed * 9
      seed = seed ^ (seed >> 4)
      seed = seed * 0x27d4eb2d
      seed = seed ^ (seed >> 15)

      -- then write out the initialized seed into the buffer
      randbuffer[tid] = seed
    end
  end
  -- compile for GPU
  rand_init_body = GPU.kernelwrap(rand_init_body)

  local blocksize   = 64
  local nblocks     = math.ceil(size_randbuffer / blocksize)
  local MAXGRIDDIM  = 65536
  local terra rand_init_launcher()
    var grid_x : uint32, grid_y : uint32, grid_z : uint32 =
        get_grid_dimensions(nblocks, MAXGRIDDIM)
    var params = terralib.CUDAParams {
      grid_x, grid_y, grid_z,
      blocksize, 1, 1,
      0, nil
    }
    rand_init_body(&params)
  end
  -- do the initial seed
  rand_init_launcher()
end

local function getseed() return currentseed end

-- very simple random number generation macro
local naiverand = macro(function(tid)
  -- make sure that we've initialized the generator
  if not randisinitialized then seedrand(currentseed) end

  -- XOR-shift PRNG from same source as above
  return quote
    var tidmod = [uint32](tid) % [uint32](size_randbuffer)
    var rng_state = randbuffer[tidmod]
    rng_state = rng_state ^ (rng_state << 13)
    rng_state = rng_state ^ (rng_state >> 17)
    rng_state = rng_state ^ (rng_state << 5)
    randbuffer[tidmod] = rng_state
  in
    rng_state
  end
end)


--[[-----------------------------------------------------------------------]]--
--[[ gpu_util Interface                                                    ]]--
--[[-----------------------------------------------------------------------]]--
GPU.kernelwrap = require 'ebb.src.cukernelwrap'

GPU.printf     = printf
GPU.block_id   = block_id
GPU.thread_id  = thread_id
GPU.global_tid = global_tid
GPU.num_blocks = num_blocks

GPU.check                   = cuda_error_checking
GPU.malloc                  = cuda_terra_malloc
GPU.free                    = cuda_terra_free
GPU.memcpy_cpu_from_gpu     = cuda_memcpy_cpu_from_gpu
GPU.memcpy_gpu_from_cpu     = cuda_memcpy_gpu_from_cpu
GPU.memcpy_gpu_from_gpu     = cuda_memcpy_gpu_from_gpu
GPU.peek_last_error         = cuda_peek_at_last_error

GPU.barrier    = macro(function() return quote cudalib.nvvm_barrier0() end end)
GPU.sync       = cuda_sync_wrapper_with_peek
--GPU.sync       = terralib.externfunction("cudaThreadSynchronize", {} -> int)
GPU.device_sync = terralib.externfunction("cudaDeviceSynchronize", {} -> int)

GPU.get_grid_dimensions = get_grid_dimensions

GPU.cbrt  = cbrt
GPU.sqrt  = sqrt
GPU.cos   = cos
GPU.acos  = acos
GPU.sin   = sin
GPU.asin  = asin
GPU.tan   = tan
GPU.atan  = atan
GPU.floor = floor
GPU.ceil  = ceil
GPU.fabs  = fabs
GPU.log   = log
GPU.pow   = pow
GPU.fmod  = fmod

GPU.fmin  = fmin
GPU.fmax  = fmax

-- Intrinsic atomic reductions:
GPU.atomic_add_float = atomic_add_float
GPU.atomic_max_int32 = reduce_max_int32
GPU.reduce_min_int32 = reduce_min_int32
GPU.reduce_add_int32 = reduce_add_int32
--GPU.reduce_and_b32   = reduce_and_b32
--GPU.reduce_or_b32    = reduce_or_b32

GPU.reduce_add_uint64   = reduce_add_uint64

-- Slow operations:
GPU.atomic_add_uint64_SLOW = generate_slow_atomic_64(add,uint64)
GPU.atomic_add_double_SLOW = generate_slow_atomic_64(add,double)

GPU.atomic_mul_float_SLOW  = generate_slow_atomic_32(mul,float)
GPU.atomic_mul_double_SLOW = generate_slow_atomic_64(mul,double)
GPU.atomic_mul_int32_SLOW  = generate_slow_atomic_32(mul,int32)
GPU.atomic_mul_uint64_SLOW = generate_slow_atomic_64(mul,uint64)

GPU.atomic_div_float_SLOW  = generate_slow_atomic_32(div,float)
GPU.atomic_div_double_SLOW = generate_slow_atomic_64(div,double)

GPU.atomic_min_uint64_SLOW = generate_slow_atomic_64(min,uint64)
GPU.atomic_min_double_SLOW = generate_slow_atomic_64(min,double)
GPU.atomic_min_float_SLOW  = generate_slow_atomic_32(min,float)

GPU.atomic_max_uint64_SLOW = generate_slow_atomic_64(max,uint64)
GPU.atomic_max_double_SLOW = generate_slow_atomic_64(max,double)
GPU.atomic_max_float_SLOW  = generate_slow_atomic_32(max,float)

-- Algorithms
GPU.ReductionObj = ReductionObj
GPU.reserve_idx  = reserve_idx

GPU.seedrand = seedrand
GPU.getseed  = getseed
GPU.rand     = naiverand
GPU.RAND_MAX  = RAND_MAX


