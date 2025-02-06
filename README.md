# NP Basin Hopping and solvent simulations

Scripts and code for basin hopping simulations of Nanoparticles with Monte Carlo (MC) moves that swap atom types of neighboring atoms. Additionally, scripts for creating solvated nanoparticles and running molecular dynamics (MD) simulations.

![basin hopping](./decahedral_basin_hopping_local_min_1080p.gif)

## Getting Started

### Dependencies

* python>=3.12
* ASE>=3.24.0 

if using MACE potentials:
* mace-torch>=0.3.9

### Usage

#### Creating simulation structures

The notebook contained in the repo demonstrates how to create solvated and ligated nanoparticles that can then be simulated by 
either basin hopping MC or MD.

#### Basin Hopping Monte Carlo Simulations
Create a conda or python virtual environment and install the required dependencies, ensuring that ASE is installed from source. 
Following this, to run the basin hopping simulations copy the files in the basin_hopping directory to ase/ase/optimize/ in the 
ASE install. A script illustrating how to run the basin hopping simulations is included in the scripts directory. Additional scripts 
for analysing surface composition are included in utils.

#### MD simulations

An example input script for running a LAMMPS simulation with MACE potential is included in the MD directory. Note to run these 
simulations requires compiling LAMMPS with MACE and additionally Kokkos for GPU acceleration. See the [MACE documentation](https://mace-docs.readthedocs.io/en/latest/guide/lammps.html) for instructions.

### DFT data
The solvated_np_pbe_data.xyz.zip contains the calculated data for 154 randomly selected local minima structures from basin hopping trajecotries of AuCu NPs with up to 192 atoms. The data was calculated with VASP 
using the PBE functional and settings that replicate that of the mptraj dataset used for the MACE-MP models (i.e. energy cutoff = 520 eV, EDIFF = 1e-6), see [here](https://github.com/ACEsuit/mace-mp).

