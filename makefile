# CXX := /opt/homebrew/opt/llvm/bin/clang++
CXX := g++
TARGET := NZD

INCLUDE_DIR := include
SRC_DIR := src
OBJ_DIR := obj

BUILD ?= release

CXXSTD := -std=c++20
WARNINGS := -Wall
INCLUDES := -I$(INCLUDE_DIR)

RELEASE_CXXFLAGS := -O3 -march=native -ffast-math -funroll-loops -DNDEBUG -fopenmp
RELEASE_LDFLAGS := -flto -fopenmpm -pthread
DEBUG_CXXFLAGS := -O0 -g -fno-omit-frame-pointer
DEBUG_LDFLAGS :=

THREAD_FLAGS := -pthread

ifeq ($(BUILD),release)
  CXXFLAGS := $(CXXSTD) $(WARNINGS) $(INCLUDES) $(RELEASE_CXXFLAGS) $(THREAD_FLAGS)
  LDFLAGS := $(RELEASE_LDFLAGS) $(THREAD_FLAGS)
else ifeq ($(BUILD),debug)
  CXXFLAGS := $(CXXSTD) $(WARNINGS) $(INCLUDES) $(DEBUG_CXXFLAGS) $(THREAD_FLAGS)
  LDFLAGS := $(DEBUG_LDFLAGS) $(THREAD_FLAGS)
else
  $(error BUILD must be 'release' or 'debug')
endif

SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking ($(BUILD))..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build finished: $(TARGET)"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling ($(BUILD)) $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(TARGET)

release:
	@$(MAKE) BUILD=release

debug:
	@$(MAKE) BUILD=debug

.PHONY: all clean release debug