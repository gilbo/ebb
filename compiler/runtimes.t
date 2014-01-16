local Runtime = {}

function Runtime:New (str)
	return setmetatable({__string=str}, self)
end

function Runtime:__tostring ()
	return self.__string
end

local singlecore  = Runtime:New('Single Core Runtime')
local gpu         = Runtime:New('GPU Runtime')

local function is_valid_runtime(rt)
	return getmetatable(rt) == Runtime
end

return {
	singlecore       = singlecore,
	gpu              = gpu,
	is_valid_runtime = is_valid_runtime
}