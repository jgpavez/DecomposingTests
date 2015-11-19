# Simple makefile
testRamdom: testRandom.cxx
	g++ -c -fPIC Root/*.cxx -I. `root-config --cflags`
	g++ -c -fPIC cMorphWrapper.cxx -I. `root-config --cflags`
	g++ -shared -Wl,-soname,libcMorphWrapper.so -o libcMorphWrapper.so *.o `root-config --glibs`
