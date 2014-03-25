local S = {}
package.loaded["soleil_liszt.fluid"] = S

import "compiler.liszt"
local Grid = terralib.require "compiler.grid"
local cmath = terralib.includecstring "#include <math.h>"

--------------------------------------------------------------------------------
--[[ Fluid type                                                             ]]--
--------------------------------------------------------------------------------

local Fluid = {}
Fluid.__index = Fluid

function S.NewFluid(gasConstant,
                    gamma,
                    dynamicViscosityRef,
                    dynamicViscosityTemperatureRef,
                    prandtl)
    local f = setmetatable({}, Fluid)

    -- member variables
    f.gasConstant = gasConstant
    f.gamma       = gamma
    f.gammaMinus1 = gamma - 1.0    
    f.dynamicViscosityRef            = dynamicViscosityRef
    f.dynamicViscosityTemperatureRef = dynamicViscosityTemperatureRef
    f.cv = gasConstant/(gamma-1)
    f.cp = gamma*f.cv
    f.prantdtl      = prandtl
    f.cpOverPrandtl = f.cp / f.prandtl

    -- member functions
    f.GetSoundSpeed = L.NewMacro(function(temperature)
        return liszt `f.gamma * f.gasConstant * temperature
    end)
    f.GetDynamicViscosity = L.NewMacro(function(temperature)
        return liszt `f.dynamicViscosityRef *
        math.pow((temperature/ f.dynamicViscosityTemperatureRef), 0.75)
    end)

    return f
end
