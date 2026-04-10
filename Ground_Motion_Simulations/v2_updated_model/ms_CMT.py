#%% import libraries

import os
import pathlib
import salvus.namespace as sn

import more_itertools
import salvus.mesh.layered_meshing as lm
from salvus.mesh.layered_meshing.detail.mesh_from_domain import mesh_from_domain
from shutil import rmtree

import numpy as np
from shutil import rmtree
import time

#%% Get current folder absolute location
current_file = pathlib.Path(__file__)
current_dir = current_file.parent
print(f"Current file: {current_file}")
print(f"Current directory: {current_dir}")

# Project setup
# Set project directory relative to current file location
PROJECT_DIR = "./png_simulations/proj_aftershock_6_7"
project_path = current_dir / PROJECT_DIR

# Check if the project directory exists
print(f"Project path: {project_path}")
print(f"Project directory exists: {project_path.exists()}")

# Uncomment if you want to create  a fresh project directory
# If the project directory exists, delete it
if project_path.exists():
    print(f"Project directory {project_path} exists. Deleting it.")
    rmtree(project_path)  # Remove the entire directory and its contents

# Slurm site "snel_cpu_slurm" to run the simulation on.
# Local site "local_cpu" to run the simulation on the local machine.
SALVUS_FLOW_SITE_NAME = os.environ.get("SITE_NAME", "local_cpu") 

# Wall time for slurm execution
wall_time_in_seconds = 172800

# Simulation parameters
reference_frequency = 0.1           # Hz
elements_per_wavelength = 8.0       # This is the default value, but can be adjusted
model_order = 4                     # This is the default value, but can be adjusted
end_time_in_seconds = 120.0         # Simulation end time
ranks_per_job = 128                 # Number of ranks per job for Slurm

#xxxx Source time function parameters: Differ per earthquake xxxx#
half_duration_in_seconds = 2.3      # Half duration of the Gaussian wavelet
time_shift_in_seconds = 8.11        # Time shift of the Gaussian wavelet

# Absorbing boundary conditions parameters
reference_velocity=4000.0           # Reference velocity for the absorbing boundary conditions
number_of_wavelengths=0.0           # Number of wavelengths for the absorbing boundary conditions


# %% Creating the domain
d = sn.domain.dim3.UtmDomain.from_spherical_chunk(
    min_latitude=-5.0,
    max_latitude=-7.0,
    min_longitude=141.0,
    max_longitude=144.0,
)

# Plot: Have a look at the domain to make sure it is correct.
# d.plot()

# %% Where to download topography data to.
topo_filename = "gmrt_topography.nc"

# Query data from the GMRT web service.
if not os.path.exists(topo_filename):
    d.download_topography_and_bathymetry(
        filename=topo_filename,
        # The buffer is useful for example when adding sponge layers
        # for absorbing boundaries so we recommend to always use it.
        buffer_in_degrees=0.1,
        resolution="default",
    )

# %% """Data is loaded to Salvus compatible SurfaceTopography object. It will resample
# to a regular grid and convert to the UTM coordinates of the domain.
# This will later be added to the Salvus project."""

t1 = sn.topography.cartesian.SurfaceTopography.from_gmrt_file(
    name="png1_topo",
    data=topo_filename,
    resample_topo_nx=200,
    # If the coordinate conversion is very slow, consider decimating.
    decimate_topo_factor=5,
    # Smooth if necessary.
    gaussian_std_in_meters=0.0,
    # Make sure to pass the correct UTM zone.
    utm=d.utm,
)

t2 = sn.topography.cartesian.SurfaceTopography.from_gmrt_file(
    name="small_avg1",
    data=topo_filename,
    resample_topo_nx=500,
    # If the coordinate conversion is very slow, consider decimating.
    decimate_topo_factor=5,
    # Smooth if necessary.
    gaussian_std_in_meters=10000.0,
    # Make sure to pass the correct UTM zone.
    utm=d.utm,
)

# Plot: Visualize topography
# t2.ds.dem.T.plot(aspect=1, size=7)
# t1.ds.dem.T.plot(aspect=1, size=7)

# Create the topography of the mesh.
_dem = (t1.ds.dem - t1.ds.dem.max()).assign_attrs(
    {"reference_elevation": float(t1.ds.dem.max())}
)

dem = _dem.copy(data=_dem)

_dem1 = (t2.ds.dem - t2.ds.dem.max()).assign_attrs(
    {"reference_elevation": float(t2.ds.dem.max())}
)

dem1 = _dem1.copy(data=_dem1)

#%% Define the 1D models for the mesh
materials = [
    lm.material.elastic.Velocity.from_params(vp=2500.0, vs=1200.0, rho=2100.0),
    # lm.material.elastic.Velocity.from_params(vp=5380.0, vs=2950.0, rho=2670.0),
    lm.material.elastic.Velocity.from_params(vp=6600.0, vs=3700.0, rho=2900.0),
    lm.material.elastic.Velocity.from_params(vp=7200.0, vs=4000.0, rho=3050.0),
    lm.material.elastic.Velocity.from_params(vp=8080.0, vs=4470.0, rho=3380.0),
    lm.material.elastic.Velocity.from_params(vp=8080.0, vs=4470.0, rho=3380.0),
]

interfaces_flat = [
    lm.interface.Surface(dem1),
    # lm.interface.Surface(dem.assign_attrs({"reference_elevation": dem.reference_elevation})),
    # lm.interface.Surface(dem1.assign_attrs({"ref": dem1.ref - 3000.0})),
    lm.interface.Hyperplane.at(dem.reference_elevation - 9000.0),
    lm.interface.Hyperplane.at(dem.reference_elevation - 18000.0),
    lm.interface.Hyperplane.at(dem.reference_elevation - 26000.0),
    lm.interface.Hyperplane.at(dem.reference_elevation - 35000.0),
    lm.interface.Hyperplane.at(dem.reference_elevation - 60000.0),
]

interfaces_topo = [
    lm.interface.Surface(dem),
    # lm.interface.Surface(dem.assign_attrs({"reference_elevation": dem.reference_elevation})),
    # lm.interface.Surface(dem.assign_attrs({"ref": dem.ref - 3000.0})),
    lm.interface.Hyperplane.at(dem.reference_elevation - 9000.0),
    lm.interface.Hyperplane.at(dem.reference_elevation - 18000.0),
    lm.interface.Hyperplane.at(dem.reference_elevation - 26000.0),
    lm.interface.Hyperplane.at(dem.reference_elevation - 35000.0),
    lm.interface.Hyperplane.at(dem.reference_elevation - 60000.0),
]
# %% Create the flat and topography layered models
# The flat model
layered_model_flat = lm.LayeredModel(
    list(more_itertools.interleave_longest(interfaces_flat, materials))
)

# The topography model
layered_model_topo = lm.LayeredModel(
    list(more_itertools.interleave_longest(interfaces_topo, materials))
)

# %% Model resolution parameters

mr = sn.MeshResolution(
    reference_frequency=reference_frequency, 
    elements_per_wavelength=elements_per_wavelength, 
    model_order=model_order
)

#%% Establishing the flat mesh
mesh_flat = mesh_from_domain(
    domain=d,
    model=lm.MeshingProtocol(
        layered_model_flat,
        interlayer_coarsening_policy=[
            lm.InterlayerConstant(),
            # lm.InterlayerDoubling(),
            lm.InterlayerDoubling(),
            lm.InterlayerDoubling(),
            lm.InterlayerDoubling(),
        ],
        ab=sn.AbsorbingBoundaryParameters(
            reference_velocity= reference_velocity,
            number_of_wavelengths= number_of_wavelengths,
            reference_frequency= reference_frequency,
        ),
    ),
    mesh_resolution=mr
)

# Plot: Visualize the flat mesh 
# mesh_flat

#%% Establsishing the mesh with the topography
mesh_topo = mesh_from_domain(
    domain=d,
    model=lm.MeshingProtocol(
        layered_model_topo,
        interlayer_coarsening_policy=[
            lm.InterlayerConstant(),
            # lm.InterlayerDoubling(),
            lm.InterlayerDoubling(),
            lm.InterlayerDoubling(),
            lm.InterlayerDoubling(),
        ],
        ab=sn.AbsorbingBoundaryParameters(
            reference_velocity=reference_velocity,
            number_of_wavelengths=number_of_wavelengths,
            reference_frequency=reference_frequency,
        ),
    ),
    mesh_resolution=mr
)

# Plot: Visualize the mesh with topography
# mesh_topo

#%% Establishing the project
p = sn.Project.from_domain(path=PROJECT_DIR, domain=d, load_if_exists=True)

# %% Define the source 
# Src / Rec reference coordinates.
src_x, src_y, src_z = 678320.92,9302914.92, -20500.0

#Source moment tensor
Mxx=-6.39E+18
Myy=-1.93E+18
Mzz=8.32E+18
Myz=5.78E+18
Mxz=8.04E+18
Mxy=-3.89E+18

src = sn.simple_config.source.cartesian.MomentTensorPoint3D(
    x=src_x,
    y=src_y,
    z=src_z,
    mxx=Mxx,
    myy=Myy,
    mzz=Mzz,
    myz=Myz,
    mxz=Mxz,
    mxy=Mxy,
)

# %% Define the receivers array
import numpy as np

rec = sn.simple_config.receiver.cartesian.collections.SideSetArrayPoint3D(
    y=np.arange(9230000, 9430000, 10000),
    x=np.arange(500000, 830000, 10000),
    depth_in_meters=0.0,
    fields=["velocity"],
)

#%% Adding the event source and receiver configurations to the project
p += sn.Event(event_name="PNG", sources=src, receivers=rec)

#%% DEfine the source time function
ec = sn.EventConfiguration(
    wavelet=sn.simple_config.stf.GaussianRate(
        half_duration_in_seconds=half_duration_in_seconds, 
        time_shift_in_seconds=time_shift_in_seconds
    ),
    waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
        end_time_in_seconds= end_time_in_seconds
    ),
)

wavelet=sn.simple_config.stf.GaussianRate(
        half_duration_in_seconds=half_duration_in_seconds, 
        time_shift_in_seconds=time_shift_in_seconds
    )

# Plot: Visualize the wavelet
wavelet.plot()

# %% Prepare the simulation flat surface

p += sn.UnstructuredMeshSimulationConfiguration(
    name="flat", unstructured_mesh=mesh_flat, event_configuration=ec
)

#%%# Prepare simulation for tensor order 2 topography

p += sn.UnstructuredMeshSimulationConfiguration(
    name="topo", unstructured_mesh=mesh_topo, event_configuration=ec
)

#%% Plot: Visualize simulation configuration
# p.viz.nb.simulation_setup("topo", ["PNG"])

# %% Launch simulation for tensor order 2
p.simulations.launch(
    "topo",
    events=["PNG"],
    site_name='local_cpu',
    # site_name = 'snel_cpu_slurm',
    # wall_time_in_seconds_per_job=wall_time_in_seconds,
    ranks_per_job=ranks_per_job,
    extra_output_configuration={
        "surface_data": {
            "sampling_interval_in_time_steps": 1,
            "fields": ["velocity"],
            "side_sets": ["z1"],
        },
    },
)
p.simulations.query(block=True)

#%% Launch simulation for tensor order 1 flat
p.simulations.launch(
    "flat",
    events=["PNG"],
    site_name='local_cpu',
    # site_name = 'snel_cpu_slurm',
    # wall_time_in_seconds_per_job=wall_time_in_seconds,
    ranks_per_job=ranks_per_job,
    extra_output_configuration={
        "surface_data": {
            "sampling_interval_in_time_steps": 1,
            "fields": ["velocity"],
            "side_sets": ["z1"],
        },
        "memory_per_rank_in_MB": 2000.0,
    },
)
p.simulations.query(block=True)

#%% Plot: Visualize time series
# p.simulations.query(block=True)
# p.viz.nb.waveforms(["flat", "topo"], receiver_field="velocity")
