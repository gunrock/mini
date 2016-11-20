# ----------------------------------------------------------------
# Gunrock -- Fast and Efficient GPU Graph Library
# ----------------------------------------------------------------
# This source code is distributed under the terms of LICENSE.TXT
# in the root directory of this source distribution.
# ----------------------------------------------------------------

#-------------------------------------------------------------------------------
# (make test) Test driver for ALGO
#-------------------------------------------------------------------------------

include ../BaseMakefile.mk

ALGO = pr
test: bin/test_$(ALGO)_$(NVCC_VERSION)_$(ARCH_SUFFIX)

bin/test_$(ALGO)_$(NVCC_VERSION)_$(ARCH_SUFFIX) : test_$(ALGO).cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(SM_TARGETS) $(OPTIONS) -o bin/test_$(ALGO)_$(NVCC_VERSION)_$(ARCH_SUFFIX) test_$(ALGO).cu $(ARCH) $(INC) --expt-relaxed-constexpr

.DEFAULT_GOAL := test
