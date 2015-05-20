--GPU-TEST
if not terralib.cudacompile then return end

import 'compiler.liszt'
L.default_processor = L.GPU
require 'examples.heattransfer'