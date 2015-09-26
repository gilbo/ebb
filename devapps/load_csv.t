import 'compiler.liszt'

-- local cells = L.NewRelation{ name = 'cells', size = 8 }
-- local cells = L.NewRelation{ name = 'cells', dims = {2,4} }
-- local cells = L.NewRelation{ name = 'cells', dims = {1,8} }
-- local cells = L.NewRelation{ name = 'cells', dims = {1,8,1} }
local cells = L.NewRelation{ name = 'cells', dims = {2,2,2} }

local liszt dump(c, field)
  L.print(c[field])
end

--------------------------------------------------------------------------------
print("\nTesting loading of scalars ...")
--------------------------------------------------------------------------------

cells:NewField('sf', L.float)
cells:NewField('sd', L.double)
cells:NewField('si', L.int)

cells.sf:LoadFromCSV('devapps/load_csv_files/sf.csv')
cells.sd:LoadFromCSV('devapps/load_csv_files/sd.csv')
cells.si:LoadFromCSV('devapps/load_csv_files/si.csv')

cells:foreach(dump, 'sf')
cells:foreach(dump, 'sd')
cells:foreach(dump, 'si')

--------------------------------------------------------------------------------
print("\nTesting loading of vectors of length 2 ...")
--------------------------------------------------------------------------------

cells:NewField('v2f', L.vec2f)
cells:NewField('v2d', L.vec2d)
cells:NewField('v2i', L.vec2i)

cells.v2f:LoadFromCSV('devapps/load_csv_files/v2f.csv')
cells.v2d:LoadFromCSV('devapps/load_csv_files/v2d.csv')
cells.v2i:LoadFromCSV('devapps/load_csv_files/v2i.csv')

cells:foreach(dump, 'v2f')
cells:foreach(dump, 'v2d')
cells:foreach(dump, 'v2i')

--------------------------------------------------------------------------------
print("\nTesting loading of matrices of length 2X3 ...")
--------------------------------------------------------------------------------

cells:NewField('m2x3f', L.mat2x3f)
cells:NewField('m2x3d', L.mat2x3d)
cells:NewField('m2x3i', L.mat2x3i)

cells.m2x3f:LoadFromCSV('devapps/load_csv_files/m2x3f.csv')
cells.m2x3d:LoadFromCSV('devapps/load_csv_files/m2x3d.csv')
cells.m2x3i:LoadFromCSV('devapps/load_csv_files/m2x3i.csv')

cells:foreach(dump, 'm2x3f')
cells:foreach(dump, 'm2x3d')
cells:foreach(dump, 'm2x3i')

--------------------------------------------------------------------------------
print("\nSaving all fields ...")
--------------------------------------------------------------------------------

cells.sf:SaveToCSV('devapps/load_csv_files/sf-saved.csv')
cells.sd:SaveToCSV('devapps/load_csv_files/sd-saved.csv')
cells.si:SaveToCSV('devapps/load_csv_files/si-saved.csv')

cells.v2f:SaveToCSV('devapps/load_csv_files/v2f-saved.csv')
cells.v2d:SaveToCSV('devapps/load_csv_files/v2d-saved.csv')
cells.v2i:SaveToCSV('devapps/load_csv_files/v2i-saved.csv')

cells.m2x3f:SaveToCSV('devapps/load_csv_files/m2x3f-saved.csv')
cells.m2x3d:SaveToCSV('devapps/load_csv_files/m2x3d-saved.csv')
cells.m2x3i:SaveToCSV('devapps/load_csv_files/m2x3i-saved.csv')
