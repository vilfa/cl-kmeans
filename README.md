# cl-kmeans

cl-kmeans is a command line tool that lets you compress images using k-means
clustering. This can be done using a single core, multiple cores using OpenMP,
or the GPU using OpenCL.

## Requirements

This project depends on OpenMP and OpenCL, so make sure to install them and any
dependencies. You should also have the appropriate OpenCL implementation for your
hardware installed. More info [here](https://wiki.archlinux.org/title/GPGPU). 
If, you're on Arch Linux, the appropriate command should be below. If not, use 
packages specific for your distro.
```bash
$ sudo pacman -S openmp opencl-headers ocl-icd opencl-(mesa|amd|nvidia)
```

You can then check you platform capabilities by running
```bash
$ clinfo
```

## Installation/Usage

Clone the repo.
```bash
$ git clone https://github.com/vilfa/cl-kmeans.git
$ cd cl-kmeans
```

Compile and run the project.
```bash
$ mkdir build && make
$ ./build/compress -iin.png -oout.png -t8 -k64 -n50
```

## License

[MIT](https://github.com/vilfa/cl-kmeans/blob/master/LICENSE)
