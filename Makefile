compile:
	gfortran src/*.f90 -shared -o lib/axisem_helpers.so
	rm *.mod
