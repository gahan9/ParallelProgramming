# Define the compiler
HIP_PATH ?= $(wildcard /opt/rocm)
HIPCC := $(HIP_PATH)/bin/hipcc

# Define GPU Architecture
GPU_ARCH := gfx942

# Define the source files
SOURCES := vector_add.cpp

# Define the object files
OBJECTS := $(SOURCES:.cpp=.o)

# Define the executable name
EXECUTABLE := ./vector_add.exe

# cmake -B <build-directory> <source-directory> -DCMAKE_PREFIX_PATH=/opt/rocm
# cmake --build <build-directory> --target all --parallel <N>

.PHONY: all clean test

# Define the Compiler flags
CXXFLAGS := -O3 -g -fopenmp --offload-arch=$(GPU_ARCH)

# Define the linker flags
LDFLAGS := -O3 -g -fopenmp --offload-arch=$(GPU_ARCH)

CXX := $(HIPCC)
# CMAKE_PREFIX_PATH=$(HIP_PATH)

# Rule to build the executable
$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@

# Rule to build object files
%.o: %.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

# Default target
all: $(EXECUTABLE) test

test:
	$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o
