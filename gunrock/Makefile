ARCH=\
	 -gencode arch=compute_35,code=compute_35

OPTIONS=-std=c++11 -ccbin=/usr/bin/g++-5 -Xcompiler="-Wundef" -O2 -g -Xcompiler="-Werror" -lineinfo  --expt-extended-lambda -use_fast_math -Xptxas="-v" -I src

test: tests/test.cu src/*.hxx src/*/*.hxx
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

clean:
	rm test
