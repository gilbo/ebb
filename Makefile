all:
	(cd runtime; make)

clean:
	(cd runtime; make clean)

test:
	./run_tests.lua