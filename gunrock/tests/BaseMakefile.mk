NVCC="$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' | sed 's/..*//'))

ARCH_SUFFIX = x86_64
ARCH = -m64

GEN_SM37 = -gencode=arch=compute_37,code=\"sm_37,compute_37\"
GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GEN_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\"
SM_TARGETS = $(GEN_SM35)

MGPU_INC= "../../../external/moderngpu/src"

INC=-I.. -I../../src -I$(MGPU_INC)

OPTIONS=-std=c++11 -ccbin=/usr/bin/g++-5 -Xcompiler="-Wundef" -O2 -g -Xcompiler="-Werror" -lineinfo  --expt-extended-lambda -use_fast_math -Xptxas="-v"

DEPS= $(wildcard ../../src/*.hxx) \
	  $(wildcard ../../src/*/*.hxx)

clean :
	rm -f bin/*_$(NVCC_VERSION)_$(ARCH_SUFFIX)*
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o
