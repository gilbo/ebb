local C = require 'compiler.c'

local function sanitize_function_name(unsafe)
  -- eliminate dots for subtables
  local safe = string.gsub(unsafe, ".", "_")
  return safe
end

-- Wrap a Terra function such that it takes no struct or array arguments by value.
-- All occurrences of struct(or array)-by-value are unpacked before being passed in
--    and then re-packed once inside the function.
return function(terrafn, verbose, annotations)
  terrafn:setinlined(true)
  --print('gonna gettype')
  local T = terrafn:gettype()
  --print('done gettype')
  assert(T.returntype:isunit(),
          "cukernelwrap: kernel must not return anything.")

  -- All the work is done here
  local function recurse(exprs, types)
    local kernelSyms = terralib.newlist()
    local unpackExprs = terralib.newlist()
    local repackExprs = terralib.newlist()
    for i,exp in ipairs(exprs) do
      local typ = types[i]
      -- Base case: primitive/pointer types can be passed in directly
      if typ:ispointer() or typ:isprimitive() then
        local sym = symbol(typ)
        kernelSyms:insert(sym)
        unpackExprs:insert(exp)
        repackExprs:insert(sym)
      -- Recursive cases
      elseif typ:isstruct() then
        local recExprs = typ.entries:map(function(e) return `exp.[e.field] end)
        local recTypes = typ.entries:map(function(e) return e.type end)
        local recKernelSyms, recUnpackExprs, recRepackExprs =
                                                recurse(recExprs, recTypes)
        kernelSyms:insertall(recKernelSyms)
        unpackExprs:insertall(recUnpackExprs)
        repackExprs:insert(`typ { [recRepackExprs] })
      elseif typ:isarray() then
        local recExprs = terralib.newlist()
        local recTypes = terralib.newlist()
        for i=1,typ.N do
          recExprs:insert(`exp[ [i-1] ])
          recTypes:insert(typ.type)
        end
        local recKernelSyms, recUnpackExprs, recRepackExprs =
                                                recurse(recExprs, recTypes)
        kernelSyms:insertall(recKernelSyms)
        unpackExprs:insertall(recUnpackExprs)
        repackExprs:insert(`array([recRepackExprs]))
      else
        error("cukernelwrap: type "..tostring(typ).." not a primitive, "..
              "pointer, struct, or array. Impossible?")
      end
    end
    return kernelSyms, unpackExprs, repackExprs
  end

  local outersyms = T.parameters:map(function(t) return symbol(t) end)
  local kernelSyms, unpackExprs, repackExprs = recurse(outersyms, T.parameters)

  -- The actual kernel takes unpacked args, re-packs them,
  --    then calls the original function passed in by the user.
  local terra kernel([kernelSyms]) : {}
    terrafn([repackExprs])
  end
  local safename = sanitize_function_name(terrafn.name)
  local modulewrapper, cudaloader = terralib.cudacompile({
      [safename]={kernel = kernel, annotations = annotations}
    },
    verbose,
    nil,  -- do not specify version
    false -- defer CUDA loading
  )
  local cudakern = modulewrapper[safename]

  -- We return a wrapper around the kernel that takes the
  --    original arguments, unpacks them, then calls the kernel.
  -- It also handles defering loading of cuda code on the first execution
  --    rather than at compile time.
  local is_loaded = global(bool, false)
  local error_buf_sz = 2048
  local terra wrapper(kernelparams: &terralib.CUDAParams, [outersyms]) : {}
    -- on the first load, make sure to load
    if not is_loaded then
      is_loaded = true
      var error_buf : int8[error_buf_sz]
      if 0 ~= cudaloader(nil,nil,error_buf,error_buf_sz) then
        C.printf("CUDA LOAD ERROR: %s\n", error_buf)
        terralib.traceback(nil)
        C.exit(1)
      end
    end

    var err = cudakern(kernelparams, [unpackExprs])
    if err ~= 0 then
      C.printf("CUDA EXEC ERROR code %d\n", err)
      C.printf("CUDA EXEC ERROR %s\n", C.cudaGetErrorString(err))
      terralib.traceback(nil)
      C.exit(1)
    end
  end
  wrapper:setinlined(true)
  return wrapper
end
