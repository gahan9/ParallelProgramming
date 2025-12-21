# Define the compiler
HIP_PATH ?= $(wildcard /opt/rocm)
HIPCC := $(HIP_PATH)/bin/hipcc
OUTPUT_DIR := profiling_output

# Define GPU Architecture
GPU_ARCH := gfx942

# Define the source files
SOURCES := vector_add.cpp
SOURCES_DYNAMIC := vector_add_dynamic.cpp

# Define the object files
OBJECTS := $(SOURCES:.cpp=.o)
OBJECTS_DYNAMIC := $(SOURCES_DYNAMIC:.cpp=.o)

# Define the executable names
EXECUTABLE := ./vector_add.exe
EXECUTABLE_DYNAMIC := ./vector_add_dynamic.exe

# cmake -B <build-directory> <source-directory> -DCMAKE_PREFIX_PATH=/opt/rocm
# cmake --build <build-directory> --target all --parallel <N>

.PHONY: all clean test test_dynamic profile profile_dynamic profile_both profile_detailed profile_kernel clean_profile clean_all

# Define the Compiler flags
CXXFLAGS := -O3 -g -fopenmp --offload-arch=$(GPU_ARCH)

# Define the linker flags
LDFLAGS := -O3 -g -fopenmp --offload-arch=$(GPU_ARCH)

CXX := $(HIPCC)
# CMAKE_PREFIX_PATH=$(HIP_PATH)

# Rule to build the baseline executable
$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@

# Rule to build the dynamic executable
$(EXECUTABLE_DYNAMIC): $(OBJECTS_DYNAMIC)
	$(HIPCC) $(OBJECTS_DYNAMIC) -o $@

# Rule to build object files
%.o: %.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

# Default target - build both executables
all: $(EXECUTABLE) $(EXECUTABLE_DYNAMIC)

# Test baseline version
test: $(EXECUTABLE)
	$(EXECUTABLE)

# Test dynamic version
test_dynamic: $(EXECUTABLE_DYNAMIC)
	$(EXECUTABLE_DYNAMIC)

# Profile baseline version
profile: $(EXECUTABLE)
	@echo "Profiling baseline vector_add.exe..."
	@mkdir -p $(OUTPUT_DIR)
	rocprofv3 --summary --sys-trace --hsa-trace --output-format csv -d $(OUTPUT_DIR)/baseline -- $(EXECUTABLE)

# Profile dynamic version
profile_dynamic: $(EXECUTABLE_DYNAMIC)
	@echo "Profiling dynamic vector_add_dynamic.exe..."
	@mkdir -p $(OUTPUT_DIR)
	rocprofv3 --summary --sys-trace --hsa-trace --output-format csv -d $(OUTPUT_DIR)/dynamic -- $(EXECUTABLE_DYNAMIC)

# Profile both versions
profile_both: $(EXECUTABLE) $(EXECUTABLE_DYNAMIC)
	@echo "Profiling both versions..."
	@mkdir -p $(OUTPUT_DIR)
	@echo "Profiling baseline..."
	rocprofv3 --summary --sys-trace --hsa-trace --output-format csv -d $(OUTPUT_DIR)/baseline -- $(EXECUTABLE)
	@echo "Profiling dynamic..."
	rocprofv3 --summary --sys-trace --hsa-trace --output-format csv -d $(OUTPUT_DIR)/dynamic -- $(EXECUTABLE_DYNAMIC)
	@echo "Profiling complete! Results in $(OUTPUT_DIR)/"

# Profile with detailed metrics (requires compatible GPU)
profile_detailed: $(EXECUTABLE_DYNAMIC)
	@echo "Profiling with detailed performance counters..."
	@mkdir -p $(OUTPUT_DIR)
	rocprofv3 --summary --stats --timestamp on --output-format csv -d $(OUTPUT_DIR)/detailed -- $(EXECUTABLE_DYNAMIC)

# Profile with kernel timing
profile_kernel: $(EXECUTABLE_DYNAMIC)
	@echo "Profiling kernel execution time..."
	@mkdir -p $(OUTPUT_DIR)
	rocprofv3 --summary --kernel-trace --output-format csv -d $(OUTPUT_DIR)/kernel -- $(EXECUTABLE_DYNAMIC)

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE_DYNAMIC)
	rm -f $(OBJECTS) $(OBJECTS_DYNAMIC)
	rm -f $(HIP_PATH)/src/*.o

clean_profile:
	rm -rf $(OUTPUT_DIR)
	rm -rf .rocprofv3

clean_all: clean clean_profile
