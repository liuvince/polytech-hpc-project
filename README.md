# High Performance Computing Projet

## Ray tracing with Hybrid MPI and OpenMP
* pathtracer_MPI_OMP.c
```
mpicc pathtracer_MPI_OMP.c -o pathtracer -lm
```

## Ray tracing with MPI
* pathtracer_MPIlignes.c
* pathtracer_MPIpixels.c
```
mpicc pathtracer_MPIlignes.c -o pathtracer -lm
mpicc pathtracer_MPIpixels.c -o pathtracer -lm
```

## Results

![ScreenShot](https://raw.github.com/liuvince/polytech-hpc-project/blob/master/image.ppm)
