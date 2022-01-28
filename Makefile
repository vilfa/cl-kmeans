CC = gcc
# CC = clang
C_OPT_DEBUG = -Wall -Wpedantic -Wextra -g
C_OPT_RELEASE = -Wall -Wpedantic -Wextra -O3
L_OPT = -lOpenCL -lm -fopenmp
L_OPT_CUDA = -L/usr/local/cuda-11.4/targets/x86_64-linux/lib -l:libOpenCL.so -lm -fopenmp
I_OPT_CUDA = -I/usr/local/cuda-11.4/targets/x86_64-linux/include
L_OPT_NSC = -L/usr/lib64 -l:libOpenCL.so.1 -lm -fopenmp
I_OPT_NSC = -I/usr/include/cuda
BUILD_DIR = build

# all: release cuda-release nsc-release
all: release

release:
	$(CC) compress.c -o $(BUILD_DIR)/compress $(C_OPT_RELEASE) $(L_OPT)

debug:
	$(CC) compress.c -o $(BUILD_DIR)/compress $(C_OPT_DEBUG) $(L_OPT)

cuda-debug:
	$(CC) compress.c -o $(BUILD_DIR)/compress_cuda $(C_OPT_DEBUG) $(I_OPT_CUDA) $(L_OPT_CUDA)

cuda-release:
	$(CC) compress.c -o $(BUILD_DIR)/compress_cuda $(C_OPT_RELEASE) $(I_OPT_CUDA) $(L_OPT_CUDA)

nsc-debug:
	$(CC) compress.c -o $(BUILD_DIR)/compress_nsc $(C_OPT_DEBUG) $(I_OPT_NSC) $(L_OPT_NSC)

nsc-release:
	$(CC) compress.c -o $(BUILD_DIR)/compress_nsc $(C_OPT_RELEASE) $(I_OPT_NSC) $(L_OPT_NSC)

clean:
	rm -f $(BUILD_DIR)/*
