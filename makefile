SRC_DIR := .
BIN_DIR := bin
BUILD_DIR := build

TARGET := deep-fryer
HEADERS := $(SRC_DIR)/kernel.cuh

NVCC := nvcc
NVCC_FLAGS := -g -G -Xcompiler -Wall $(HEADERS)

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
