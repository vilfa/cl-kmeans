CC = gcc
COPT_DEBUG = -Wall -Wpedantic -Wextra
COPT_RELEASE = $(COPT_DEBUG) -O3
LOPT = -lOpenCL -lm -fopenmp
BUILD_DIR = build

all: debug

release:
	$(CC) $(COPT_RELEASE) $(LOPT) compress_cpu.c -o $(BUILD_DIR)/compress_cpu

debug:
	$(CC) $(COPT_DEBUG) $(LOPT) compress_cpu.c -o $(BUILD_DIR)/compress_cpu

clean:
	rm -f $(BUILD_DIR)/*
