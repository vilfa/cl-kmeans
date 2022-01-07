CC = gcc
COPT_DEBUG = -Wall -Wpedantic -Wextra -g
COPT_RELEASE = -Wall -Wpedantic -Wextra -O3
LOPT = -lOpenCL -lm -fopenmp
BUILD_DIR = build

all: release

release:
	$(CC) $(COPT_RELEASE) $(LOPT) compress.c -o $(BUILD_DIR)/compress

debug:
	$(CC) $(COPT_DEBUG) $(LOPT) compress.c -o $(BUILD_DIR)/compress

clean:
	rm -f $(BUILD_DIR)/*
