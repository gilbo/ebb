package.path = package.path .. ";./compiler/?.lua;./compiler/?.t"
require "liszt"

local Parser = terralib.require('terra/tests/lib/parsing')

local lisztlanguage = {
   name        = "liszt", -- name for debugging
   entrypoints = {"liszt_kernel"},
   keywords    = {"var"},

   expression = function(self, lexer)
      local kernel_ast = Parser.Parse(lang, lexer, "liszt_kernel")
      --[[ this function is called in place of executing the code that 
           we parsed 
      --]]
      return function () return kernel_ast end
   end
}

return lisztlanguage
