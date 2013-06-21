package.path = package.path .. ";./compiler/?.lua;./compiler/?.t"

-- Import liszt parser as a local module
-- (keep liszt language internals out of global environment for liszt user)
local liszt = require "liszt"
_G.liszt             = nil
package.loaded.liszt = nil

local Parser = terralib.require('terra/tests/lib/parsing')

local lisztlanguage = {
   name        = "liszt", -- name for debugging
   entrypoints = {"liszt_kernel"},
   keywords    = {"var", "foreach"},

   expression = function(self, lexer)
      local kernel_ast = Parser.Parse(liszt.lang, lexer, "liszt_kernel")
      --[[ this function is called in place of executing the code that 
           we parsed 
      --]]

      return function () 
         return kernel_ast
      end
   end
}

return lisztlanguage
