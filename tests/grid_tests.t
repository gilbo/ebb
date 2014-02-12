import "compiler.liszt"
LDB = terralib.require "compiler.ldb"
Grid = terralib.require "compiler.grid"
local print = L.print

function addSource(dim, srcGrid, dstGrid, index, dt)
    for i = 1, dim[1] + 2 do
        for j = 1, dim[2] + 2 do
            local x = dstGrid.get({i, j})
            local y = srcGrid.get({i, j})
            local z = {x[1], x[2], x[3], x[4], x[5]}

            z[index] = x[index] + y[index] * dt

            dstGrid.set({i, j}, z)
--            dstGrid.set({i, j}, {x[1], x[2], x[3], x[4], x[5] + y[5] * dt})
        end
    end
end
function setBoundary(dim, grid, index, boundaryFlag)
    for i = 2, dim[1] + 1 do
        -- 1
        local x = grid.get({1, i})
        local y = grid.get({2, i})
        local z = {x[1], x[2], x[3], x[4], x[5]}

        if boundaryFlag == 1 then
            z[index] = -y[index]
        else
            z[index] = y[index]
        end

        grid.set({1, i}, z)

        -- 2
        local x = grid.get({dim[1] + 2, i})
        local y = grid.get({dim[1] + 1, i})
        local z = {x[1], x[2], x[3], x[4], x[5]}

        if boundaryFlag == 1 then
            z[index] = -y[index]
        else
            z[index] = y[index]
        end

        grid.set({dim[1] + 2, i}, z)

        -- 3
        local x = grid.get({i, 1})
        local y = grid.get({i, 2})
        local z = {x[1], x[2], x[3], x[4], x[5]}

        if boundaryFlag == 2 then
            z[index] = -y[index]
        else
            z[index] = y[index]
        end

        grid.set({i, 1}, z)

        -- 4
        local x = grid.get({i, dim[2] + 2})
        local y = grid.get({i, dim[2] + 1})
        local z = {x[1], x[2], x[3], x[4], x[5]}

        if boundaryFlag == 2 then
            z[index] = -y[index]
        else
            z[index] = y[index]
        end

        grid.set({1, i}, z)
    end

    -- 1
    local x = grid.get({1, 1})
    local y = grid.get({2, 1})
    local z = grid.get({1, 2})
    local w = {x[1], x[2], x[3], x[4], x[5]}

    w[index] = 0.5 * (y[index] + z[index])

    grid.set({1, 1}, w)

    -- 2
    local x = grid.get({1, dim[2] + 2})
    local y = grid.get({2, dim[2] + 2})
    local z = grid.get({1, dim[2] + 1})
    local w = {x[1], x[2], x[3], x[4], x[5]}

    w[index] = 0.5 * (y[index] + z[index])

    grid.set({1, dim[2] + 2}, w)

    -- 3
    local x = grid.get({dim[1] + 2, 1})
    local y = grid.get({dim[1] + 1, 2})
    local z = grid.get({dim[1] + 2, 2})
    local w = {x[1], x[2], x[3], x[4], x[5]}

    w[index] = 0.5 * (y[index] + z[index])

    grid.set({dim[1] + 2, 1}, w)

    -- 4
    local x = grid.get({dim[1] + 2, dim[2] + 2})
    local y = grid.get({dim[1] + 1, dim[2] + 2})
    local z = grid.get({dim[1] + 2, 2})
    local w = {x[1], x[2], x[3], x[4], x[5]}

    w[index] = 0.5 * (y[index] + z[index])

    grid.set({dim[1] + 2, dim[2] + 2}, w)
end

function advect(dim, srcGrid, dstGrid, index, uGrid, uIndex, vGrid, vIndex, dt, boundaryFlag)
    local dt0 = dt * (dim[1] + 1)

    for i = 2, dim[1] + 1 do
        for j = 2, dim[2] + 1 do
            -- x
            local u0 = uGrid.get({i, j})
            local x = i - dt0 * u0[uIndex]
 
            if x < 0.5 then
                x = 0.5
            end

            if x > dim[1] + 0.5 then
                x = dim[1] + 0.5
            end

            local i0 = math.floor(x)
            local i1 = i0 + 1

            -- y
            local v0 = vGrid.get({i, j})
            local y = j - dt0 * v0[vIndex]

            if y < 0.5 then
                y = 0.5
            end

            if y > dim[2] + 0.5 then
                y = dim[2] + 0.5
            end

            local j0 = math.floor(x)
            local j1 = j0 + 1

            -- ...
            local s1 = x - i0
            local s0 = 1 - s1
            local t1 = y - j0
            local t0 = 1 - t1

            local d = dstGrid.get({i, j})
            local d00 = srcGrid.get({i0, j0})
            local d01 = srcGrid.get({i0, j1})
            local d10 = srcGrid.get({i1, j0})
            local d11 = srcGrid.get({i1, j1})
            local z = {d[1], d[2], d[3], d[4], d[5]}

            z[index] = s0 * (t0 * d00[index] + t1 * d01[index]) + s1 * (t0 * d10[index] + t1 * d11[index])

            dstGrid.set({i, j}, z)
        end
    end

    setBoundary(dim, dstGrid, boundaryFlag)
end

function diffuse(dim, srcGrid, dstGrid, diff, dt, index, boundaryFlag)
    local a = dt * diff * (dim[1] + 1) * (dim[2] + 1)

    for k = 1, 20 do
        for i = 2, dim[1] + 1 do
            for j = 2, dim[2] + 1 do
                local x = dstGrid.get({i, j})
                local y = srcGrid.get({i, j})
                local z = {x[1], x[2], x[3], x[4], x[5]}

                local leftX = dstGrid.get({i - 1, j})
                local rightX = dstGrid.get({i + 1, j})
                local botX = dstGrid.get({i, j - 1})
                local topX = dstGrid.get({i, j + 1})

                z[index] = (y[index] + a * (leftX[index] + rightX[index] + botX[index] + topX[index])) / (1 + 4 * a)
                dstGrid.set({i, j}, z)
            end
        end

        setBoundary(dim, grid, index, boundaryFlag)
    end
end

function project(dim, uGrid, uIndex, vGrid, vIndex, pGrid, pIndex, divGrid, divIndex)
    local h = 1 / (dim[1] + 1)

    for i = 2, dim[1] + 1 do
        for j = 2, dim[2] + 1 do
            local x = divGrid.get({i, j})
            local y = {x[1], x[2], x[3], x[4], x[5]}

            local leftU = uGrid.get({i - 1, j})
            local rightU = uGrid.get({i + 1, j})

            local botV = vGrid.get({i, j + 1})
            local topV = vGrid.get({i, j - 1})

            y[divIndex] = -0.5 * h * (rightU[uIndex] - leftU[uIndex] + botV[vIndex] - topV[vIndex])
            divGrid.set({i, j}, y)

            local z = pGrid.get({i, j})
            local w = {z[1], z[2], z[3], z[4], z[5]}

            w[pIndex] = 0
            pGrid.set({i, j}, w)
        end
    end

    setBoundary(dim, divGrid, divIndex, 0)
    setBoundary(dim, pGrid, pIndex, 0)

    for k = 1, 20 do
        for i = 1, dim[1] do
            for j = 1, dim[2] do
                local x = pGrid.get({i, j})
                local y = {x[1], x[2], x[3], x[4], x[5]}

                local div = divGrid.get({i, j})

                local leftP = pGrid.get({i - 1, j})
                local rightP = pGrid.get({i + 1, j})

                local botP = pGrid.get({i, j + 1})
                local topP = pGrid.get({i, j - 1})

                y[pIndex] = 0.25 * (div[divIndex] + rightP[pIndex] + leftP[pIndex] + botP[pIndex] + topP[pIndex])
                pGrid.set({i, j}, y)
            end
        end

        setBoundary(dim, pGrid, pIndex, 0)
    end

    for i = 1, dim[1] do
        for j = 1, dim[2] do
            local u = uGrid.get({i, j})
            local u0 = {u[1], u[2], u[3], u[4], u[5]}

            local leftP = pGrid.get({i - 1, j})
            local rightP = pGrid.get({i + 1, j})

            u0[uIndex] = (rightP[pIndex] - leftP[pIndex]) / h
            uGrid.set({i, j}, u0)

            local v = vGrid.get({i, j})
            local v0 = {v[1], v[2], v[3], v[4], v[5]}

            local botP = pGrid.get({i, j + 1})
            local topP = pGrid.get({i, j - 1})

            v0[vIndex] = (botP[pIndex] - topP[pIndex]) / h
            vGrid.set({i, j}, v0)
        end
    end

    setBoundary(dim, uGrid, uIndex, 1)
    setBoundary(dim, vGrid, vIndex, 2)
end

function dens_step(dim, srcGrid, dstGrid, index, uGrid, uIndex, vGrid, vIndex, diff, dt)
    addSource(dim, srcGrid, dstGrid, dt)
    diffuse(dim, dstGrid, srcGrid, diff, dt, index, 0)
    advect(dim, srcGrid, index, dstGrid, index, uGrid, uIndex, vGrid, vIndex, dt, 0)
end

function vel_step(dim, uSrcGrid, uDstGrid, uIndex, vSrcGrid, vDstGrid, vIndex, visc, dt)
    addSource(dim, uSrcGrid, uDstGrid, uIndex, dt)
    addSource(dim, vSrcGrid, vDstGrid, vIndex, dt)

    diffuse(dim, uDstGrid, uSrcGrid, diff, dt, uIndex, 1)
    diffuse(dim, vDstGrid, vSrcGrid, diff, dt, vIndex, 2)

    project(dim, uDstGrid, uIndex, vDstGrid, vIndex, uSrcGrid, uIndex, vSrcGrid, vIndex)
    advect(dim, uSrcGrid, uIndex, uDstGrid, uIndex, vSrcGrid, vIndex, uSrcGrid, uIndex, dt, 1)
    advect(dim, vSrcGrid, vIndex, vDstGrid, vIndex, vSrcGrid, vIndex, uSrcGrid, uIndex, dt, 2)
    project(dim, uDstGrid, uIndex, vDstGrid, vIndex, uSrcGrid, uIndex, vSrcGrid, vIndex)
end

-- Cell = {v_x, v_y, v_z, pressure, density}

local 2dfluid = Grid.initUniformGrid({4, 4}, {1, 1, 1, 1, 1}, {{key = 'viscosity', value = 0.0}, {key = 'diffusion', value = 0.0}})

local 2dfluidPrev = Grid.initUniformGrid({4, 4}, {1, 1, 1, 1, 1}, {{key = 'viscosity', value = 0.0}, {key = 'diffusion', value = 0.0}})

--addSource({0, 0}, localGrid, localGrid, 3, 5)

dt = 0.008

vel_setp({2, 2}, 2dfluid.GridClass, 2dfluidPrev.GridClass, 2dfluid['viscosity'], dt)
dens_step({2, 2}, 2dfluid.GridClass, 2dfluidPrev.GridClass, 2dfluid['diffusion'], dt)

localGrid:print()

--[[
local center_of_cell_1 = terra (mem : &vector(float, 3), i : uint)
    mem[0] = vectorof(float, 0.5, 0.5, 0.5)
end

M.particles.position:LoadFromCallback(center_of_cell_1)
M:updateParticles()

M.particles.cell:print()

(liszt kernel(p in M.particles)
    p.position = {1.5, 0.5, 0.5}
end)()
M:updateParticles()    

M.particles.cell:print()
]]


