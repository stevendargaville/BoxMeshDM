This code builds unstructured meshes for a 2D square [0,1]x[0,1] in parallel with MPI. It relies on PETSc configured with triangle and hdf5 (``--download-triangle --download-hdf5``).

### Executable

To build an executable which can be called from the terminal make sure ``PETSC_DIR`` and ``PETSC_ARCH`` environmental variables are set and then call ``make clean && make``. 

There are four variables that can be enabled from the command line, ``-target_edge_length 0.0025``, ``-final_smooth_its 2``, ``-write_mesh true`` and ``-print_stats true`` which are set by default. For large scale parallel testing you probably want to set ``-write_mesh false``.

To visualise the mesh, enable ``-write_mesh true``, then on the command line run ``${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py box_mesh.h5``. The resulting ``.xmf`` file can be visualised in Paraview with the XDMF reader.

### Library

Rather than building an executable, the code can be compiled as a library. Hence rather than writing out the mesh at scale, the routine ``GenerateBoxMeshDM`` can be called directly from existing code as it returns a parallel, load balanced PETSc DM that can be used without I/O. 

Ensure ``PETSC_DIR`` and ``PETSC_ARCH`` environmental variables are set and then call ``make clean && make lib`` and then you need to include the ``.h`` file in your code and link to the output ``libbox_2D_gen_unstruc_mesh``. 

#### Notes

This should be used to generate large meshes, it is not robust when the number of elements per MPI rank is small (say <100k).

The meshes generated should be independent of the number of MPI ranks used.