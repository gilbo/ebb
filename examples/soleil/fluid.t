local S = {}
package.loaded["soleil_liszt.fluid"] = S

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
    f.gasConstant = gasConstant
    f.gamma       = gamma
    f.gammaMinus1 = gamma - 1.0    
    f.dynamicViscosityRef            = dynamicViscosityRef
    f.dynamicViscosityTemperatureRef = dynamicViscosityTemperatureRef
    f.cv = gasConstant/(gamma-1)
    f.cp = gamma*f.cv
    f.prantdtl      = prandtl
    f.cpOverPrandtl = f.cp / f.prandtl
    return f
end

function Fluid:GetSoundSpeed(temperature)
    return math.sqrt(self.gamma * self.gasConstant * temperature)
end

function Fluid:GetDynamicViscosity(temperature)
    return self.dynamicViscosityRef *
           math.pow((temperature/ self.dunamicViscosityTemperatureRef),
                    0.75)
end
