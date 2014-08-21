compile:
	gfortran src/*.f90 -g -shared -o lib/axisem_helpers.so
	rm *.mod
