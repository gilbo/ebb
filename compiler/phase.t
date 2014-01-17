local Phase = {}
Phase.__index = Phase
function Phase.new (string) return setmetatable({string=string}, Phase) end
function Phase:__tostring () return self.string end

local phases = {}
phases.READ_ONLY       = Phase.new("<Read>")
phases.REDUCE_PLUS     = Phase.new("<Additive Reduction>")
phases.REDUCE_MULTIPLY = Phase.new("<Multiplicative Reduction>")
phases.REDUCE_MIN      = Phase.new("<MIN Reduction>")
phases.REDUCE_MAX      = Phase.new("<MAX Reduction>")
phases.REDUCE_AND      = Phase.new("<AND Reduction>") -- AND/OR/XOR by assignment type
phases.REDUCE_OR       = Phase.new("<OR Reduction>")
phases.WRITE_ONLY      = Phase.new("<Write>")

phases['+']    = phases.REDUCE_PLUS
phases['-']    = phases.REDUCE_PLUS
phases['*']    = phases.REDUCE_MULTIPLY
phases['/']    = phases.REDUCE_MULTIPLY
phases['min']  = phases.REDUCE_MIN
phases['max']  = phases.REDUCE_MAX
phases['and']  = phases.REDUCE_AND
phases['or']   = phases.REDUCE_OR

------------------------------------------------------------------------------
--[[ Phase Dictionary                                                     ]]--
------------------------------------------------------------------------------
--[[ Keeps track of Field accesses and field access metadata:
     * whether the field was read, written, or reduced
     * where the Access occurred (for debugging/error reporting)
--]]
local FieldAccessRecord = {}
FieldAccessRecord.__index = FieldAccessRecord

function FieldAccessRecord.new (field, phase, node)
    return setmetatable({
        phase      = phase,
        field      = field,
        linenumber = node.linenumber,
        filename   = node.filename,
        offset     = node.offset,
        name       = node.name
    }, FieldAccessRecord)
end

local PhaseDict = {}
PhaseDict.__index = PhaseDict

function PhaseDict.new(diag)
    return setmetatable({
        diag=diag, 
        dict={}, 
        ctxt={phases.READ_ONLY}}, PhaseDict)
end

function PhaseDict:enterLhs (reduce_op)
    local ph = phases[reduce_op] or phases.WRITE_ONLY
    self.ctxt[#self.ctxt+1] = ph
end
function PhaseDict:enterRhs ()
    self.ctxt[#self.ctxt+1] = phases.READ_ONLY
end
function PhaseDict:leaveLhs ()
    local sz = #self.ctxt
    local ph = self.ctxt[sz]
    self.ctxt[sz] = nil
    -- assert that popped phase is a write or reduce phase
    assert(ph and ph ~= phases.READ_ONLY)
end
function PhaseDict:leaveRhs ()
    local sz = #self.ctxt
    local ph = self.ctxt[sz]
    self.ctxt[sz] = nil
    -- assert that popped phase is read only
    assert(ph == phases.READ_ONLY)    
end
function PhaseDict:store (field, node)
    local phase = self.ctxt[#self.ctxt]
    -- DEBUG print detected fields
    -- local fn = tostring(field.owner._name) .. '.' .. field.name
    -- print("Found field " .. fn .. ' in phase ' .. tostring(phase) .. ' at line: ' .. tostring(node.linenumber))
    if not self.dict[field] then
        self.dict[field] = FieldAccessRecord.new(field, phase, node)
        return true
    elseif phase ~= self.dict[field].phase then
        local rec = self.dict[field]
        local fn  = tostring(field.owner._name) .. '.' .. field.name
        self.diag:reporterror(node, "access of '"..fn.."' field in "..tostring(phase).. 
            ' phase conflicts with earlier access in '..tostring(rec.phase)..
            ' phase at '..rec.filename..':'..tostring(rec.linenumber))
        return false
    end
    return true
end

local P = {
    PhaseDict       = PhaseDict,
    READ_ONLY       = phases.READ_ONLY,
    REDUCE_PLUS     = phases.REDUCE_PLUS,
    REDUCE_MULTIPLY = phases.REDUCE_MULTIPLY,
    REDUCE_MIN      = phases.REDUCE_MIN,
    REDUCE_MAX      = phases.REDUCE_MAX,
    REDUCE_AND      = phases.REDUCE_AND,
    REDUCE_OR       = phases.REDUCE_OR,
    WRITE_ONLY      = phases.WRITE_ONLY     
}

return P
