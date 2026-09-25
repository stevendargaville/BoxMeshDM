// =========================================================
// Main Driver for the BoxMeshDM executable
// Kept out of BoxMeshDM.cpp so the library object never defines main
// =========================================================
#include "BoxMeshDM.h"
#include <iostream>
#include <cstdlib>
#include <petscviewerhdf5.h>

int main(int argc, char** argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);    

    // Parse command line options
    double target_len = 0.0025;
    PetscBool set;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-target_edge_length", &target_len, &set));

    PetscBool write_mesh = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-write_mesh", &write_mesh, NULL));

    PetscBool integrity_check = PETSC_TRUE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-integrity_check", &integrity_check, NULL));    

    PetscBool print_stats = PETSC_TRUE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_stats", &print_stats, NULL));    

    PetscInt final_smooth_its = 4;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-final_smooth_its", &final_smooth_its, &set));
    int final_smooths = final_smooth_its;

    double domain_width = 1.0;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-domain_width", &domain_width, &set));
    
    double domain_height = 1.0;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-domain_height", &domain_height, &set));

    PetscInt agglomeration_factor = 1;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-agglomeration_factor", &agglomeration_factor, &set));
    int agglom_factor = agglomeration_factor;

    // Generate the DMPlex for this mesh
    DM dm = GenerateBoxMeshDMAgglom(MPI_COMM_WORLD, target_len, domain_width, domain_height, final_smooths, integrity_check, print_stats, agglom_factor);

    // Check a valid mesh has been generated
    if (dm) {

        // Write output if requested
        // Can view this in paraview with:
        // /home/sdargavi/projects/dependencies/petsc_main/lib/petsc/bin/petsc_gen_xdmf.py box_mesh.h5
        // then using the XDMF reader with:
        // paraview box_mesh.xmf
        if (write_mesh) {
#ifdef PETSC_HAVE_HDF5         
           PetscViewer viewer;
           if (comm_rank == 0 && print_stats) {
                 std::cout << "Writing out mesh...\n";        
           }
           PetscCall(PetscViewerHDF5Open(MPI_COMM_WORLD, "box_mesh.h5", FILE_MODE_WRITE, &viewer));
           PetscCall(DMView(dm, viewer));
           PetscCall(PetscViewerDestroy(&viewer));     
#else 
           if (comm_rank == 0) {
               std::cerr << "-write_mesh not available without HDF5 enabled in PETSc.\n";
           } 
#endif
         }

        PetscCall(DMDestroy(&dm));
    } else {
        PetscCall(PetscFinalize());
        return EXIT_FAILURE;
    }

    PetscCall(PetscFinalize());
    return 0;
}
