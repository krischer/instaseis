# Poor man's Makefile...
compile:
	gfortran -c src/global_parameters.f90
	gfortran -c src/finite_elem_mapping.f90
	mkdir -p lib
	gfortran src/*.f90 -shared -o lib/axisem_helpers.so
	rm *.mod *.o
