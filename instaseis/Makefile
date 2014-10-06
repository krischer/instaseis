# Poor man's Makefile...
compile:
	gfortran -c src/global_parameters.f90
	gfortran -c src/finite_elem_mapping.f90
	gfortran -c src/lanczos.f90
	mkdir -p lib
	gfortran src/*.f90 -fPIC -shared -o lib/instaseis_helpers.so
	rm *.mod *.o
