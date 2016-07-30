
# Run the appropriate command in here to build a shared object for tutorial 11

dummy:
	# don't do anything

osx_11:
	gcc -shared -std=c99 -o 11___c_obj.so -I../../include 11___c_code.c

linux_11:
	gcc -shared -std=c99 -o 11___c_obj.so -I../../include 11___c_code.c

# Location of Terra and Ebb directories
# These may have to be edited
TERRA_RELEASE:=../../terra
EBB_INCLUDE:=../../include

osx_18:
	c++ -Wall -g -fPIC -I $(TERRA_RELEASE)/include -I $(EBB_INCLUDE) \
      18_c_embedding.c -o 18_exec \
      -lcurses -lz -L $(TERRA_RELEASE)/lib -lterra \
      -pagezero_size 10000 -image_base 100000000

linux_18:
	c++ -Wall -g -fPIC -I $(TERRA_RELEASE)/include -I $(EBB_INCLUDE) \
      18_c_embedding.c -o 18_exec \
      -lcurses -lz -L $(TERRA_RELEASE)/lib -lterra \
      -ldl -pthread

clean:
	-rm 11___c_obj.so
	-rm sample_out.off
	-rm 18_exec
	-rm -rf 18_exec.dSYM # debug sym directory on mac
