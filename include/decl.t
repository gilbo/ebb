
--[[
    The DECL file holds declarations of objects primarily to break
    cyclic dependencies.  However, it'd probably be good practice
    to go ahead and place anything which could be considered a
    "declaration"
        (i.e. here is a thing in the system which exists and may
            be user facing)
    in this file.
]]--
local DECL = {}


DECL.C = terralib.includecstring [[
    #include <stdlib.h>
    #include <string.h>
    #include <stdio.h>
    #include <math.h>

    FILE *get_stderr () { return stderr; }
]]


local function make_prototype(tb)
    tb.__index = tb
    return tb
end

DECL.LRelation  = make_prototype({})
DECL.LField     = make_prototype({})
DECL.LScalar    = make_prototype({})
DECL.LVector    = make_prototype({})
DECL.LMacro     = make_prototype({})

--  We use the convention that one should refer to the
-- current metatable to determine what an object is.
-- While this is isn't quite duck-typing, it does ensure
-- that passing one of the following checks is meaningful.
-- Namely, if an object passes one of the following checks
-- it must have all methods installed on the above prototype.
--  This is a safer idea than checking arbitrary table entries
-- e.g. obj.is_relation or obj.kind == 'relation' b/c it is
-- much harder to spoof a metatable.
--  Ideally in a prototypal object system we could simply ask
-- is X a prototype of Y and have a routine check the entire
-- prototype chain.  The below is a compromise from such a scheme
-- which doesn't require quite as much trouble to implement.
-- We could replace it at some point.
function DECL.is_relation (obj) return getmetatable(obj) == DECL.LRelation end
function DECL.is_field (obj)    return getmetatable(obj) == DECL.LField end
function DECL.is_scalar (obj)   return getmetatable(obj) == DECL.LScalar end
function DECL.is_vector (obj)   return getmetatable(obj) == DECL.LVector end
function DECL.is_macro (obj)    return getmetatable(obj) == DECL.LMacro end



return DECL

