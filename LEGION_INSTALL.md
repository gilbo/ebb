
# Legion Installation

*PLEASE make sure to contact the Liszt-Ebb developers before trying to install for Legion.*  This documentation is only meant to _help_ with that process, and may be entirely incorrect.  You are unlikely to succeed without direct support.

## Outline of Steps

1)  Install Terra from source

2)  Verify Terra Build

3)  Build and Verify Liszt-Ebb

4)  Download Legion

5)  Re-build and Verify Liszt-Ebb

6)  Distributed Legion Installation (in-progress)



# 1 - Install Terra from source

Please go to terralang.org.  From there, go to the github page and clone the master branch to a local directory.  This doc will refer to that location as `<terra>` from here on out.  Follow the installation from source instructions provided by Terra.  Note in particular that this will require obtaining a copy of LLVM.  We use 3.5.x, so please use that version if possible.  



# 2 - Verify Terra Build

Once Terra is built, `cd tests` and execute `./run` to run Terra's test suite.  You may find that one or two tests fail (e.g. avxhadd or sgemm something something).  If that looks like your situation, you are... probably ok.  However, please notify the developers ASAP of which tests do not pass at this stage, even if you decide to press forward with your installation.  (REMEMBER, if you are not in contact with the developers at this point, you are in for a world of pain.)



# 3 - Build and Verify Liszt-Ebb

Presumably you've already cloned this git repository.  Good Job!  The rest of this document will refer to the liszt-ebb root directory as `<ebb>`.

If you just try to run `make` in `<ebb>` you will probably notice that we try to download a binary version of Terra.  IF THAT HAPPENS, THAT IS INCORRECT.  Please read the beginning of the Liszt-Ebb `<ebb>/Makefile` present in this same directory.  After skipping the copyright notice, you'll see a block that looks something like

```
# If any of the below defaults are not right for your configuration,
# Create a Makefile.inc file and set the variables in it
-include Makefile.inc

# location of terra directory in binary release, or the /release subdir in
# a custom build of Terra
TERRA_DIR?=../terra
# The following variables are only necessary for Legion development
LEGION_DIR?=
TERRA_ROOT_DIR?=
# You need LLVM installed.  If your installation of LLVM is not on the
# path, then you can point this variable at the config tool.
LLVM_CONFIG ?= $(shell which llvm-config-3.5 llvm-config | head -1)
```

Do not edit these variables here.  Instead, we're going to create a new file named `<ebb>/Makefile.inc`.  Notice that we need to provide the location of Terra via the variable `TERRA_DIR`.  Simply add the following line to `Makefile.inc`

```
TERRA_DIR=<terra>/release
```

(random info: The release directory holds the same contents as a binary distribution of Terra.)

Now the Liszt-Ebb Makefile should be able to locate Terra without problem.  Go ahead and run `make` in `<ebb>` and then after that run `./runtests`.  If any of these tests fail *TELL THE DEVELOPERS*.  None of these tests should fail.



# 4 - Download Legion

Just download it.  Clone the git directory.  We'll refer to the path to the legion root directory from here on out as `<legion>`.

Don't build it.  It's fine.  The Legion build system is complicated and we had to work around it in strange ways.



# 5 - Re-build and Verify Liszt-Ebb

`cd <ebb>` and then run `make clean`.  Before we rebuild for Legion, we need to modify our `Makefile.inc` to add two more paths:

```
LEGION_DIR=<legion>
TERRA_ROOT_DIR=<terra>
```

This will tell the makefile where all of the Terra and Legion source code is.  Great.  That should be all you need to do.

Now, just re-type `make` and the Legion runtime libraries should be built as well.  This can take a while, so using the option `make -j8` will parallelize the build process across 8 processors. (You can adjust this parameter for your machine)

To verify that Ebb continues to work with Legion, you can now run `./runtests --legion`.  If any tests here fail, please notify the developers.



# 6 - Distributed Legion Installation

Hopefully at this point, the general pattern is clear.  We can supply needed build parameters and locations to the Liszt-Ebb build process via the `Makefile.inc` file and we should try to run the test suite at each step to verify that we haven't bungled up something about the installation.

## mpicc

Make sure you have the command `mpicc` on your path.  If you don't, install MPI or modify your setup accordingly.

Uhh, on sapling, this seems to be possible by adding the following to your `.bashrc`:

```
module load mpi/openmpi/1.8.2
module load gasnet/1.22.4-openmpi
module load cuda/6.5
```

## Download GASnet

The Legion team is hosting a copy of the gasnet code on their github account.  You can make a copy of it with the command

```
git clone git@github.com:StanfordLegion/gasnet.git
```

We'll call the root directory for gasnet `<gasnet>`

## Build GASnet

```
cd <gasnet>
make CONDUIT=ibv
```

Hopefully that works out alright.  If not, you may want to just blow away the gasnet directory, re-download and rebuild.  There isn't a `make clean` option.  I also have no idea how to test whether this installation was correct.

## Modify `Makefile.inc`

Add the line

```
GASNET_DIR=<gasnet>/release
```

## Rebuild Liszt-Ebb

```
make clean
make
```

## Weird way to run?

```
mpirun -n 2 -H n0000,n0001 -npernode 1 -bind-to none ./ebb ...
```
















