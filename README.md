
# Ebb

Ebb is a (domain-specific) language for writing physical simulations.  It is part of the Liszt project at Stanford.


## Quick Setup

Once you've got your local copy of this repository, you can simply type

```
make
```

and Terra (dependency) will be downloaded for you.  When this process is done, you can type

```
./runtests
```

to make sure everything is working.  You should be good to go.


### Troubleshooting Quick Setup

If you don't have wget or unzip installed, you may run into trouble with the automatic download of Terra.  Please install those tools or try installing Terra yourself.

(You may also run into trouble if you don't have libcurses and libz installed.  If this is the case, please report back to the developers---we currently don't believe this will ever happen.)

## Windows Setup

At this time, we haven't focused on having a Windows version of Ebb.  If you're interested in helping port Ebb (there's a relatively small amount of platform-specific code) please contact the developers.


## Longer Setup Instructions

If you are working on multiple DSLs using Terra and want to avoid a redundant Terra install, you can configure the variable `TERRA_DIR` at the top of the [`Makefile`](Makefile) to locate your Terra install directory instead.  If you have a binary download, simply point `TERRA_DIR` variable at the root directory.  If you are building Terra from source, then point `TERRA_DIR` at the `release` subdirectory.  By default, `TERRA_DIR=../terra/release`.

You will still need to run `make` even if you already have your own Terra install.  Doing so will build the Ebb interpreter, which is needed to run Ebb programs.

## VDB Setup

We use a simple tool called VDB to do lightweight visualization during development of Ebb programs.  You can download this tool separately, but to simplify things, we've included a Makefile rule to download and build VDB for you.  Just run:

```
make vdb
```


## Using Ebb from C code

A [tutorial on embedding Ebb](examples/tutorials/18_c_embedding.c) into a C program is available for guidance.  Doing this embedding mostly consists of a standard embedding of the Lua/Terra interpreter.  If you find yourself having trouble with this variation on using Ebb, please contact the developers.



## More Details

See the [full manual](docs/manual.md) for more information.

## Examples

See the [examples](examples) directory for example Ebb programs.  This is a good way to get a few ideas about how to proceed once you've got some code running.



## Tests

As mentioned before, you can run the testing suite by executing
```
./run_tests
```



## Running Ebb on the GPU

To run an Ebb program on the GPU instead of CPU, simply add the command line flag.

```
ebb -g my_program.t
```

Support for simultaneous CPU/GPU use is currently being worked on.  Please contact the developers if the feature is particularly important for you.









