SRC_DIR := .
LIB_DIR := libs
BIN_DIR := bin
BUILD_DIR := build

TARGET := deep-fryer
HEADERS := -I$(SRC_DIR)/kernel.cuh
LIBS := -I$(LIB_DIR)/CImg -I$(LIB_DIR)/argparse

NVCC := nvcc
NVCC_FLAGS := -g -G -std=c++20 -Xcompiler -Wall $(HEADERS) $(LIBS)

all:	init $(TARGET)

init:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

$(TARGET):	$(BUILD_DIR)/main.o $(BUILD_DIR)/kernel.o
	$(NVCC) $^ -o $(BIN_DIR)/$@

$(BUILD_DIR)/main.o:	$(SRC_DIR)/main.cpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/kernel.o:	$(SRC_DIR)/kernel.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	-rm -f $(BUILD_DIR)/* $(BIN_DIR)/* ./*.pdb
