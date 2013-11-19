TERRADIR = "../terra"

all:    terra
	make -C runtime;

terra:
	ln -s $(TERRADIR) $@
    
clean:
	make -C runtime clean
	rm -f terra
	
test:   all
	terra/terra run_tests.lua