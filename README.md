# fft fluid solver toy
### Controls
* Click and drag to swirl fluid around
* Space + Click for an explosion
* Press up/down to double/halve viscosity   
* Press R to randomize velocity field
* Press S to toggle slow motion mode
* Press 0 to zero velocity field
* Press V to change view

![gif](/img/product.gif)

I've been wanting to implement a 2d fluid sim like this for some [other projects](https://github.com/collebrusco/gunpowder) and of course the computational fluid dynamics rabbit hole is deep.       
I found [this paper by Jos Stam (2001)](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/jgt01.pdf) that details a relatively simple FFT-based technique.   

Here I've implemented it with [MIT's FFTW](https://www.fftw.org/) FFT library, rendered the fluid field and added some controls for playing with it.      
For a school project, a team and I implemented our own FFTs in CUDA! Check that [here](https://github.com/collebrusco/361C-term-project), if you have an NVIDIA GPU this makes the solver faster.         
There are two renderers that you can switch between or overlay: one simply renders the x and y coords of each vector in the red and blue components of color, and the other renders the vectors as lines.       
The color renderer surprisingly has a very neat and fluid look to it, while the lines give you the traditional vector field rendering.

# Build and Run
My gfx projects all build on any OS with CMake, but it's easier with Mac and Linux. Windows just needs more help getting a compiler and build tools like make. More on that in flgl's [guide](https://github.com/collebrusco/flgl/blob/main/user/README.md).     
1. You will need [CMake](https://cmake.org/), install it from the link or with homebrew / apt with `brew install cmake` (mac) or `sudo apt install cmake` (linux)      
2. There is one dependency for the FFTs, [fftw](https://www.fftw.org/). install with `brew install fftw` (mac) or `sudo apt install fftw` (linux)
3. Clone the repo, cd there
```bash
git clone --recurse-submodules git@github.com:collebrusco/fluid-solver-toy; cd fluid-solver-toy
```
If you didn't clone with `--recurse-submodules` you can cd there and run `git submodule update --init --recursive`       
4. build and run with `make -j r`!     

if you're curious, that does this:
```
# builds it
mkdir -p build
cd build
cmake ..
make
# and if you include the r, runs it
./build/fluid
```

# How Does it Work?

## The Solver
The solver algorithm itself is found in [StamFFT_FluidSolver.cpp](https://github.com/collebrusco/fluid-solver-toy/blob/absfft/src/StamFFT_FluidSolver.cpp), which you can reference to see un-interrupted code. Below I show the broad strokes of the solver function. First, the input forces are simply applied to the field, shown below.
```c++
    for ( i=0 ; i<N*N ; i++ ) {
        u[i] += dt*u0[i]; u0[i] = u[i];
        v[i] += dt*v0[i]; v0[i] = v[i];
    }
```
Next, the advection scheme is performed. The paper describes this scheme as "semi-Lagrangian".
```c++
    for ( i=0 ; i<N ; i++ ) {
        for ( j=0 ; j<N ; j++ ) {
            x = i-dt*u0[i+N*j]*N; y = j-dt*v0[i+N*j]*N;
            i0 = __floor(x); s = x-i0; i0 = (N+(i0%N))%N; i1 = (i0+1)%N;
            j0 = __floor(y); t = y-j0; j0 = (N+(j0%N))%N; j1 = (j0+1)%N;
            u[i+N*j] = (1-s)*((1-t)*u0[i0+N*j0]+t*u0[i0+N*j1])+
                          s *((1-t)*u0[i1+N*j0]+t*u0[i1+N*j1]);
            v[i+N*j] = (1-s)*((1-t)*v0[i0+N*j0]+t*v0[i0+N*j1])+
                          s *((1-t)*v0[i1+N*j0]+t*v0[i1+N*j1]);
        }
    }
```
Next, in preperation for the spatial -> fourier domain transform, the 2D velocities are copied into the real part of the buffers on which we will perform the transform. `BUFF_R` and `BUFF_I` are macros to index the real and imaginary parts of the buffers.
```c++
    for ( i=0 ; i<N ; i++ ) {
        for ( j=0 ; j<N ; j++ ) {
            BUFF_R(u0, i, j) = u[i+N*j]; 
            BUFF_R(v0, i, j) = v[i+N*j];
            BUFF_I(u0, i, j) = 0.; 
            BUFF_I(v0, i, j) = 0.;
        }
    }
```
Here is the use of the FFT. The FFT forward/inverse calls are the CUDA kernels that will be discussed later. They are also abstracted so that a generic solver can be used (e.g. I use a library FFT to run the simulator on MacOS). After transforming to the fourier domain, a low-pass filter with a cutoff that is a function of viscosity is applied. At the same time, the fourier-domain vectors are projected to be tangent to concentric rings around the origin. The paper goes into more detail on how this removes the divergent component of the field. Afterwards, the fluid is transformed back into the spatial domain, and normalized. This completes a step of the solver.
```c++
    fftu->forward(); fftv->forward();

    for (int i = 0; i < N; i++) {
        x = (i <= N/2) ? i : (float)i - (float)N;
        for (int j = 0; j < N; j++) {
            y = (j <= N/2) ? j : (float)j - (float)N;
            r = x*x + y*y;
            if (r == 0.0) continue;

            float *uf = &(u0[2*(i + N*j)]);
            float *vf = &(v0[2*(i + N*j)]);
            
            float f = exp(-r * dt * visc);
            
            float ur = f * ( (1 - x*x/r)*uf[0] - x*y/r * vf[0] );
            float ui = f * ( (1 - x*x/r)*uf[1] - x*y/r * vf[1] );
            float vr = f * ( -y*x/r * uf[0] + (1 - y*y/r)*vf[0] );
            float vi = f * ( -y*x/r * uf[1] + (1 - y*y/r)*vf[1] );

            (&(u0[2*(i + N*j)]))[0] = ur;
            (&(u0[2*(i + N*j)]))[1] = ui;
            (&(v0[2*(i + N*j)]))[0] = vr;
            (&(v0[2*(i + N*j)]))[1] = vi;
        }
    }

    fftu->inverse(); fftv->inverse();

    f = 1.0/(N*N);
    for ( i=0 ; i<N ; i++ ) {
        for ( j=0 ; j<N ; j++ ) { 
            u[i+N*j] = f*BUFF_R(u0, i, j); v[i+N*j] = f*BUFF_R(v0, i, j); 
        }
    }

```
## The Renderer
The fluid simulator is not much good if you can't observe it! Moreover, I wanted to build an interactive demo of the solver that allows the user to swirl the fluid around. For this, I've used my [graphics library](https://github.com/collebrusco/flgl) that I've been maintaining for a few years. This library is based on OpenGL & GLFW, and includes a number of other common abstractions and tools. There is more to say about this library, but all that's relevant here is that it reads very similarly to and can be mixed with plain OpenGL.       
I built two renderers for the field. Both can be seen in the gif at the top of the page. The first simply places the x and y components of the vector into the red and blue color channels. This gives a surprisingly fluid like render. The renderer for this is ineffecient, but effective. I maintain a buffer of floating point x and y values, normalized to be between 0 and 1. This is buffered to the GPU every frame as a texture. The code for this (found [here](https://github.com/collebrusco/fluid-solver-toy/blob/absfft/src/rgFieldRenderer.cpp), using my flgl library) is below.
```c++
// upload sizeof(float)*2*n*n buffer to GPU
field_tex.bind();
field_tex.alloc(GL_RG32F, n, n, GL_RG, GL_FLOAT, field.buff());
field_tex.unbind();
// render texture to screen size quad
field_tex.bind();
field_tex.bind_to_unit(0);
field_shad.bind();
field_shad.uInt("uTexslot", 0);
gl.draw_mesh(quad);
field_tex.unbind();
field_shad.unbind();
```
The second renderer is the 'classic' vector field render style. The code can be found [here](https://github.com/collebrusco/fluid-solver-toy/blob/absfft/src/vecFieldRenderer.cpp). This is done by building a vertex buffer of lines originating from each vector pointing in their direction and scaled down by some constant. A line is only added every 4 vectors so that the final product is less messy. The CPU-side buffer is populated, sent to the GPU, and drawn every frame. This is much faster than the color renderer, though still requires high GPU bandwidth.
```c++

// build mesh
size_t bufi = 0;
for (size_t j = 0; j < (n); j+=4) {
    for (size_t i = 0; i < (n); i+=4) {

        // create vector vertices
        glm::vec2 vec = glm::vec2(u[i + j * n], v[i + j * n]);
        float l = glm::length(vec);
        vec *= (1./l);
        l = sqrt(l+1) - 1;
        l *= 2;
        l = max(l-0.1,0.);
        vec *= l;
        if (l<=0.) continue;

        // Scale positions to [-1, 1]
        buffer[bufi].pos = glm::vec2(-1.0f + size * i, -1.0f + size * j);

        // Calculate and scale vector
        buffer[bufi].vec = buffer[bufi].pos + coeff * vec; // Adjust vector by the given coefficient and add to position

        // Add line indices (this ultimately doesn't need this, should really be vbo only)
        size_t index = 2*bufi;
        indices.push_back(index); // start of line (position)
        indices.push_back(index+1); // end of line (position + vector)
        bufi++;
    }
}

// upload mesh to GPU
mesh.vao.bind();
mesh.vbo.bind();
mesh.vbo.buffer_data(bufi, buffer);
mesh.ibo.bind();
mesh.ibo.buffer(indices);
mesh.vao.unbind();
mesh.vbo.unbind();
mesh.ibo.unbind();
glLineWidth(2.0f);
glEnable(GL_LINE_SMOOTH);

// draw
line_shad.bind();
gl.draw_mesh(mesh, GL_LINES);
```
