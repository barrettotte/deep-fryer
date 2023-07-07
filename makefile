NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all:	deep-frier

deep-frier:	main.o kernel.o
	$(NVCC) $^ -o $@

main.o:	main.c kernel.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

kernel.o:	kernel.cu kernel.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
