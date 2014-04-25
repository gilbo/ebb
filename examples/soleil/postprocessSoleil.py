"""
Program: plotSoleil
Language: Python

    Copyright (C) 2014 Ivan Bermejo-Moreno

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


Description:
- Plot output of Soleil program, consisting of a slice containing 
the flow density field and superimpose particles as circles with center
based on their x, y coordinates and scaled by their diameter

Usage:
$ python ../plotParticles.py \
    inputFileNamePrefix \
    timeStep \
    sliceIndex \
    particlesSizeFactor \
    particlesArrowFactor \
    outputFileNamePrefix
"""

import matplotlib
import pylab
import matplotlib.pyplot as plt
import numpy
import sys
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
from matplotlib import rc
from scipy import interpolate


# -----------------------------------------------------------------------------
def readColumns(inputFileName, 
                columns = "all",
                headerLines = 0,
                ignoreCommentedLines = True,
                commentCharacter = '#',
                ignoreBlankLines = True,
                outputFormat = str):

    """ Reads and returns a given set of columns from an ASCII file 

    Input arguments:
      inputFileName: ASCII file containing the data
   
    Optional input arguments:
      columns: list of indeces of columns to be read from file (start with 1)
               of "all" to read all columns
      ignoreComentedLines: if commentCharacter is found at the beginning of
                           the line, this line will be skipped
      commentedCharacter: starting character to consider a line to be commented
      ignoreBlankLines: blank lines will be ignored
   
    Returns: columns
    """


    headerColumns = None

    if columns == "all":
#       Get number of columns from file itself
#       Open file
        inputFile = open(inputFileName, "r")
        lineCounter = 0
        for line in inputFile.readlines():
            lineCounter = lineCounter + 1
            if lineCounter <= headerLines:
#               Skip header line
                continue
            if ignoreBlankLines and len(line) == 1:
#               Skip blank line
                continue
            if ignoreCommentedLines and line[0] == commentCharacter:
#               Skip line with comment
                continue
            lineColumns = line.split()
            columns = range(1,len(lineColumns)+1)
            break
#       Close file
        inputFile.close()
    elif columns == "allWithHeaderInfo":
#       Get number of columns from file itself
#       assuming the first line is the header containing the
#       field info
#       Open file
        inputFile = open(inputFileName, "r")
        lineCounter = 0
#       Read header line
        headerLine = inputFile.readline()
#       Split header line and ignore first element (comment)
        headerColumns = headerLine.split()[1:]
        columns = range(1,len(headerColumns)+1)
#       Close file
        inputFile.close()
    elif type(columns) is not list:
        print "Columns specification", columns, "not implemented"

       
    numberOfColumnsToRetrieve = len(columns)
    if numberOfColumnsToRetrieve < 1:
        print "No columns were specified"
        
#   Initialize x, y lists
    outputColumns = list([] for element in columns)

#   Check if headerColumns exist
    if headerColumns is not None:
        for idx in range(numberOfColumnsToRetrieve):
            outputColumns[idx].append(headerColumns[columns[idx]-1])

#   Open file
    inputFile = open(inputFileName, "r")
    lineCounter = 0
    for line in inputFile.readlines():
        lineCounter = lineCounter + 1
        if lineCounter <= headerLines:
#           Skip header line
            continue
        if ignoreBlankLines and len(line) == 1:
#           Skip blank line
            continue
        if ignoreCommentedLines and line[0] == commentCharacter:
#           Skip line with comment
            continue
        lineColumns = line.split()
        if len(lineColumns) < numberOfColumnsToRetrieve:
            print "Line", lineCounter, "does not contain ", \
                   str(numberOfColumnsToRetrieve), "columns"
        for idx in range(numberOfColumnsToRetrieve):
            outputColumns[idx].append(outputFormat(lineColumns[columns[idx]-1]))

#   Close file
    inputFile.close()
            
    return outputColumns


# -----------------------------------------------------------------------------
def readValuesFromLine(inputFileName,
                       lineNumber,
                       startColumn = 1,
                       outputFormat = str):

    """ Reads and returns the list of values  

    Input arguments:
      inputFileName: ASCII file containing the data
      lineNumber: line number (starting from 1) within file
      startColumn: column number (starting from 1) to start reading values
   
    Returns: list of values in line from column startColumn until the last one
    """

#   Initialize x, y lists
    values = []

#   Open file
    inputFile = open(inputFileName, "r")
    lineCounter = 0
    for line in inputFile.readlines():
        lineCounter = lineCounter + 1
        if lineCounter == lineNumber:
#           Keep this line
            lineString = line

#   Close file
    inputFile.close()

    columns = map(outputFormat,lineString.split()[startColumn-1:])
            
    return columns



# -----------------------------------------------------------------------------
def readMatrix(inputFileName,
               headerLines = 1,
               startColumnInHeader = 3):

    """ Reads matrix from ASCII file

    Input arguments:
      inputFileName: ASCII file containing the data
      headerLines: number of header lines
      startColumnInHeader: column in header line where y values start
   
    Returns: lists of x, y and F values.
    """



#   Read header line containing y coordinates
#   assuming that the first column of the header line is
#   the comment character and the second is a label
#   Sample file:
#     -----------------
#     |# x1\x2 0 1 2 3|
#     |0 1 1 1 1      |
#     |1 2 2 2 2      |
#     |2 3 3 3 3      |
#     -----------------
    y = map(float,
            readValuesFromLine(inputFileName,
                               lineNumber = headerLines,
                               startColumn = startColumnInHeader))
#   Read rest of file
    columns = readColumns(inputFileName,
                          columns = "all",
                          headerLines = headerLines)
#   Assign 1st column to x1 coordinates
    x = map(float,columns[0])
#   Assign successive columns to F elements
    xDim=len(x)
    yDim=len(y)
    F = []
    for jdx in range(1,yDim+1):
       subF = map(float,columns[jdx])
       F.append(subF)

    return x, y, F


# --------------------------------------------------------------

inputFileNamePrefix  = sys.argv[1]
timeStep             = sys.argv[2]
fieldName            = sys.argv[3]
sliceIndex           = sys.argv[4]
particlesSizeFactor  = float(sys.argv[5])
particlesArrowFactor = float(sys.argv[6])
outputFileNamePrefix = sys.argv[7]


# Flow
inputFileNameCoordinates = inputFileNamePrefix + "_centerCoordinates.txt"
#inputFileNameField       = inputFileNamePrefix + "_dummyfield1.txt"
inputFileNameField = inputFileNamePrefix + "_" + str(timeStep) + \
                     "_flow_" + fieldName + ".txt"
with open(inputFileNameCoordinates, 'r') as f:
  xSize, ySize, numPoints, numComponents = \
    [int(elem) for elem in f.readline().split()[1:]]

coordinatesList = numpy.loadtxt(inputFileNameCoordinates,
                                usecols = (0,1,2),
                                skiprows = 1)

xCoor = numpy.array(coordinatesList)[:xSize,1]
yCoor = numpy.array(coordinatesList)[::xSize,2]

with open(inputFileNameField, 'r') as f:
  xSizeField, ySizeField, numPointsField, numComponentsField = \
    [int(elem) for elem in f.readline().split()[1:]]
fieldArray = numpy.array(numpy.loadtxt(inputFileNameField,
                                skiprows = 1))

# Check sizes match
if ( xSize != xSizeField or \
     ySize != ySizeField ):
    print "ERR: sizes differ"
    exit(1)

# Reshape fieldList into 2D array format
if numComponentsField == 1:
    fieldArray = fieldArray[:,1:].reshape(ySize,xSize)
else:
    fieldArray = fieldArray[:,1:].reshape(ySize,xSize,numComponentsField)
zeroPadding=8


# Particles
particlesPositionInputFileName = inputFileNamePrefix + "_" + \
  str(timeStep) + "_particles_position.txt"
particlesVelocityInputFileName = inputFileNamePrefix + "_" + \
  str(timeStep) + "_particles_velocity.txt"
particlesTemperatureInputFileName = inputFileNamePrefix + "_" + \
  str(timeStep) + "_particles_temperature.txt"
particlesDiameterInputFileName = inputFileNamePrefix + "_" + \
  "particles_diameter.txt"
with open(particlesPositionInputFileName, 'r') as f:
  numParticles, numComponentsField = \
    [int(elem) for elem in f.readline().split()[1:]]
particlesX, particlesY = numpy.loadtxt(particlesPositionInputFileName,
                                usecols = (1,2), unpack = True,
                                skiprows = 1)
particlesVelocityX, particlesVelocityY = numpy.loadtxt(particlesVelocityInputFileName,
                                usecols = (1,2), unpack = True,
                                skiprows = 1)
particlesDiameter = numpy.loadtxt(particlesDiameterInputFileName,
                                usecols = (1,), unpack = True,
                                skiprows = 1)
particlesTemperature = numpy.loadtxt(particlesTemperatureInputFileName,
                                usecols = (1,), unpack = True,
                                skiprows = 1)
print fieldArray
print particlesTemperature

xCoor = numpy.array(coordinatesList)[:xSize,1]
yCoor = numpy.array(coordinatesList)[::xSize,2]

fieldToPlot = numpy.array(fieldArray)
coorXToPlot = numpy.array(xCoor)
coorYToPlot = numpy.array(yCoor)


interp='nearest'
#interp='bilinear'

fig = plt.figure(figsize=(8,6))
fig.suptitle(fieldName + ' field at slice ' + str(sliceIndex) + ' with particles')
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(coorXToPlot.min(), coorXToPlot.max()),
                     ylim=(coorYToPlot.min(), coorYToPlot.max()))
ax.grid()
vmin = fieldToPlot.min()
vmax = fieldToPlot.max()

norm = colors.Normalize(vmin = vmin, vmax = vmax)
im = image.NonUniformImage(ax, interpolation=interp,
                     cmap='jet', norm = norm)
im.set_data(coorXToPlot,coorYToPlot,
            fieldToPlot)
ax.images.append(im)
ax.set_xlim(coorXToPlot.min(),coorXToPlot.max())
ax.set_ylim(coorYToPlot.min(),coorYToPlot.max())
ax.set_title(interp)
evolutionText = ax.text(0.02, 0.90, '', transform=ax.transAxes)

# Particles as circles at {X, Y} locations scaled by their diameter
particlesPlot = \
  plt.scatter(particlesX,particlesY,
              s=particlesSizeFactor*particlesDiameter,
              alpha=0.8,
              c=particlesTemperature)
plt.colorbar(im)
plt.colorbar(particlesPlot)

# Particles velocity as quiver
particlesVelocityPlot = \
  plt.quiver(particlesX, particlesY,
             particlesVelocityX, particlesVelocityY,
             particlesTemperature,
             scale=particlesArrowFactor,
             alpha=0.8)

# Write output file
outputFileName = outputFileNamePrefix + "_" + \
                 str(timeStep).zfill(zeroPadding) + \
                 "_" + fieldName + ".png"
print "Writing", outputFileName
plt.savefig(outputFileName)
