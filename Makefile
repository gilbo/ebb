TERRADIR = "../terra"

CUDPP=cudpp-2.1
CUDPP_TAR=$(CUDPP).tar.gz
CUDPP_URL=https://github.com/cudpp/cudpp/archive/2.1.tar.gz

all: terra external/$(CUDPP)
	make -C runtime;

external/$(CUDPP): external/$(CUDPP_TAR)
	tar -xf external/$(CUDPP_TAR) -C external/

external/$(CUDPP_TAR):
ifeq ("$(wildcard external)","")
	mkdir external
endif

ifeq ($(UNAME), Darwin)
	curl $(CUDPP_URL) -o external/$(CUDPP_TAR)
else
	wget $(CUDPP_URL) -O external/$(CUDPP_TAR)
endif

terra:
	ln -s $(TERRADIR) $@
    
clean:
	make -C runtime clean
	rm -rf external/$(CUDPP)
	rm -f terra
	
test: all
	terra/terra run_tests.lua