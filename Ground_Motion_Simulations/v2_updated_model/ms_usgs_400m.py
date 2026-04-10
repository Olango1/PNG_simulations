#%% import libraries
# Step 1: Import the required libraries
print("Step 1:  Import the required libraries")

import os
from os.path import exists
import pathlib
import salvus.namespace as sn

import more_itertools
import salvus.mesh.layered_meshing as lm
from salvus.mesh.layered_meshing.detail.mesh_from_domain import mesh_from_domain
from salvus.mesh.data_structures.unstructured_mesh.unstructured_mesh import UnstructuredMesh

import numpy as np
from shutil import rmtree
from time import time
import h5py
import rasterio
from scipy.interpolate import griddata
from rasterio.transform import Affine
from rasterio.crs import CRS
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Get current folder absolute location
# Step2:  Define the project folder and the main simulation parameters
print("Step 2: Define the project folder and the main simulation parameters")
current_file = pathlib.Path(__file__)
current_dir = current_file.parent
home_png_simulation = '/home/ifadel/Publications/2025_PNG/png_simulations/03_simul_cr2_1hw_2m/'
print(f"Current file: {current_file}")
print(f"Current directory: {current_dir}")

# Fault file path
fault_csv_path = "/home/ifadel/Publications/2025_PNG/emerald_files/usgs_ffm_400m.csv"
print(f"Fault CSV path: {fault_csv_path}")

# Project setup
# remote site to run the simulation on.
# snel_cpu_slurm, snel_h100_gpu_slurm, snel_a100_gpu_slurm
# local_cpu, local_gpu
SALVUS_FLOW_SITE_NAME = os.environ.get("SITE_NAME", "snel_h100_gpu_slurm")

# Wall time for slurm execution
wall_time_in_seconds = 2 * 60 * 60
ranks = 8
mesh_overwrite = False

# Topography parameters
buffer_in_degrees = 0.1
resample_topo_topo = 200
decimate_topo_factor = 5
gaussian_std_in_meters_topo = 0.0

resample_topo_flat = 200  # Change from 500
decimate_flat_factor = 5
gaussian_std_in_meters_flat = 10000.0

# Simulation parameters
reference_frequency = 1           # Hz
mesh_append = 'hp_1s_1hw_2m'
proj_append = "mash_usgs_400m_"
elements_per_wavelength = 1.5       # This is the default value, but can be adjusted
model_order = 2                     # This is the default value, but can be adjusted

end_time_in_seconds = 120.0         # Simulation end time
sampling_interval_in_time_steps = 10  # Sampling interval in time steps

# Absorbing boundary conditions parameters
absorbing_reference_frequency = 2
reference_velocity=4000           # Reference velocity for the absorbing boundary conditions
number_of_wavelengths=0           # Number of wavelengths for the absorbing boundary conditions

# # Source time function parameters for point source only
# half_duration_in_seconds = 2.3      # Half duration of the Gaussian wavelet
# time_shift_in_seconds = 8.11        # Time shift of the Gaussian wavelet

print("Simulation parameters:")
print(f"  Reference frequency: {reference_frequency} Hz")
print(f"  Mesh append: {mesh_append}")
print(f"  Elements per wavelength: {elements_per_wavelength}")
print(f"  Model order: {model_order}")
print(f"  End time: {end_time_in_seconds} s")
print(f"  Sampling interval: {sampling_interval_in_time_steps} time steps")
print(f"  Mesh overwrite: {mesh_overwrite}")
print(f"  Site name: {SALVUS_FLOW_SITE_NAME}")
print(f"  Number of ranks: {ranks}")

# Set project directory relative to current file location
project_path = home_png_simulation + "proj_" + proj_append + mesh_append

# Check if the project directory exists
print(f"Project path: {project_path}")
print(f"Project directory exists: {exists(project_path)}")

#%% Check if the project directory exists
if exists(project_path):
    print(f"Project directory {project_path} exists. Deleting it.")
    rmtree(project_path)  # Remove the entire directory and its contents


# If the project directory exists, delete it
if exists(project_path):
    print(f"Project directory {project_path} exists. Deleting it.")
    rmtree(project_path)  # Remove the entire directory and its contents

# The flat and topo mesh paths
flat_mesh_path = f'{home_png_simulation}/flat_mesh_{mesh_append}.h5'
topo_mesh_path = f'{home_png_simulation}/topo_mesh_{mesh_append}.h5'

# print flat and topo mesh paths
print(f"Flat mesh path: {flat_mesh_path}")
print(f"Topo mesh path: {topo_mesh_path}")

# If mesh overwrite is enabled and the flat mesh exists, delete it
if mesh_overwrite and os.path.exists(flat_mesh_path):
    print("Mesh overwrite is enabled. Deleting existing flat mesh.")
    os.remove(flat_mesh_path)
    os.remove(flat_mesh_path.replace('.h5', '.xdmf'))

# If mesh overwrite is enabled and the topo mesh exists, delete it
if mesh_overwrite and os.path.exists(topo_mesh_path):
    print("Mesh overwrite is enabled. Deleting existing topo mesh.")
    os.remove(topo_mesh_path)
    os.remove(topo_mesh_path.replace('.h5', '.xdmf'))

# %% Creating the domain
# Step 3: Create the simulation domain
print("Step 3: Create the simulation domain")

# Creating the domain for chen model should look something like this
d = sn.domain.dim3.UtmDomain.from_spherical_chunk(
    min_latitude=-7.2,
    max_latitude=-4.8,
    min_longitude=141.0,
    max_longitude=145.0,
)

# Have a look at the domain to make sure it is correct.
d.plot()

#%% Where to download topography data to.
# Step 4: Get the topography
print("Step 4: Get the topography")

topo_filename = "gmrt_topography.nc"

# Query data from the GMRT web service.
if not os.path.exists(topo_filename):
    d.download_topography_and_bathymetry(
        filename=topo_filename,
        # The buffer is useful for example when adding sponge layers
        # for absorbing boundaries so we recommend to always use it.
        buffer_in_degrees=buffer_in_degrees,
        resolution="default",
    )

t_flat = sn.topography.cartesian.SurfaceTopography.from_gmrt_file(
    name="png_flat",
    data=topo_filename,
    resample_topo_nx=resample_topo_flat,
    # If the coordinate conversion is very slow, consider decimating.
    decimate_topo_factor=decimate_flat_factor,
    # Smooth if necessary.
    gaussian_std_in_meters=gaussian_std_in_meters_flat,
    # Make sure to pass the correct UTM zone.
    utm=d.utm,
)

t_topo = sn.topography.cartesian.SurfaceTopography.from_gmrt_file(
    name="png_topo",
    data=topo_filename,
    resample_topo_nx=resample_topo_topo,
    # If the coordinate conversion is very slow, consider decimating.
    decimate_topo_factor=decimate_topo_factor,
    # Smooth if necessary.
    gaussian_std_in_meters=gaussian_std_in_meters_topo,
    # Make sure to pass the correct UTM zone.
    utm=d.utm,
)

#%% Plot: Visualize topography
t_flat.ds.dem.T.plot(aspect=1, size=7, cmap="terrain")
t_topo.ds.dem.T.plot(aspect=1, size=7, cmap="terrain")

# # Make a new variable with the flat topography 
# # streteched to the maximum and minimum elevation of the topo topography
# t_flat_stretched = t_flat.ds.dem.copy(
#     data=t_flat.ds.dem * (t_topo.ds.dem.max() - t_topo.ds.dem.min()) / (t_flat.ds.dem.max() - t_flat.ds.dem.min())
# )     

# t_flat_stretched.T.plot(aspect=1, size=7, cmap="terrain")

#%% Create the topography of the mesh.

_dem_flat = (t_flat.ds.dem).assign_attrs(
    {"reference_elevation": 0.0}
    # {"reference_elevation": float(t_flat.ds.dem.max())}
)
dem_flat = _dem_flat.copy(data=_dem_flat)
plt.figure()
dem_flat.plot()
plt.show()
plt.close()

_dem_topo = (t_topo.ds.dem).assign_attrs(
    {"reference_elevation": 0.0}
    # {"reference_elevation": float(t_topo.ds.dem.max())}
)
dem_topo = _dem_topo.copy(data=_dem_topo)
plt.figure()
dem_topo.plot()
plt.show()
plt.close()

# Plot the two topographies and the difference between them in one figure
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
dem_flat.T.plot(ax=ax[0], cmap="terrain")
dem_topo.T.plot(ax=ax[1], cmap="terrain")
(dem_topo - dem_flat).T.plot(ax=ax[2], cmap="bwr")
plt.tight_layout()
plt.show()
plt.close()

#%% Define the 1D models and create the mesh
# Step 5: Define the 1D model and create the mesh
print("Step 5: Define the 1D model and create the mesh")

materials = [
    lm.material.elastic.Velocity.from_params(vp=2500.0, vs=1200.0, rho=2100.0),
    lm.material.elastic.Velocity.from_params(vp=4000.0, vs=2400.0, rho=2400.0),
    lm.material.elastic.Velocity.from_params(vp=6000.0, vs=3500.0, rho=2700.0),
    lm.material.elastic.Velocity.from_params(vp=6600.0, vs=3700.0, rho=2900.0),
    lm.material.elastic.Velocity.from_params(vp=7200.0, vs=4000.0, rho=3050.0),
    lm.material.elastic.Velocity.from_params(vp=8080.0, vs=4470.0, rho=3380.0),
]

interfaces_flat = [
    lm.interface.Hyperplane.at(dem_flat.reference_elevation + dem_topo.max().values),
    lm.interface.Surface(dem_flat.assign_attrs({"reference_elevation": 
                         dem_flat.reference_elevation - 1000.0})),
    lm.interface.Surface(dem_flat.assign_attrs({"reference_elevation": 
                         dem_flat.reference_elevation - 2000.0})),
    lm.interface.Hyperplane.at(dem_topo.reference_elevation - 18000.0),
    lm.interface.Hyperplane.at(dem_topo.reference_elevation - 26000.0),
    lm.interface.Hyperplane.at(dem_topo.reference_elevation - 35000.0),
    lm.interface.Hyperplane.at(dem_topo.reference_elevation - 60000.0),
]

interfaces_topo = [
    lm.interface.Surface(dem_topo),
    lm.interface.Surface(dem_topo.assign_attrs({"reference_elevation": 
                         dem_topo.reference_elevation - 1000.0})),
    lm.interface.Surface(dem_topo.assign_attrs({"reference_elevation": 
                         dem_topo.reference_elevation - 2000.0})),
    lm.interface.Hyperplane.at(dem_topo.reference_elevation - 18000.0),
    lm.interface.Hyperplane.at(dem_topo.reference_elevation - 26000.0),
    lm.interface.Hyperplane.at(dem_topo.reference_elevation - 35000.0),
    lm.interface.Hyperplane.at(dem_topo.reference_elevation - 60000.0),
]

print("Creating flat and topography layered models.")

# The flat model
layered_model_flat = lm.LayeredModel(
    list(more_itertools.interleave_longest(interfaces_flat, materials))
)

# The topography model
layered_model_topo = lm.LayeredModel(
    list(more_itertools.interleave_longest(interfaces_topo, materials))
)

# Define mesh resolution parameters and create meshes
mr = sn.MeshResolution(
    reference_frequency=reference_frequency, 
    elements_per_wavelength=elements_per_wavelength, 
    model_order=model_order
)

# Establishing the mesh with the flat topography
print("Creating flat mesh.")
tstart = time()
layer_coarsening_policy = [
                lm.InterlayerConstant(),
                lm.InterlayerDoubling(),
                lm.InterlayerDoubling(),
                lm.InterlayerDoubling(),
                lm.InterlayerDoubling(),
            ]


# Check if the flat mesh exists, other wise create it.
# For the flat mesh we do not need a constatnt layers 
# because they keep the fine detials of the topography but 
# for the flat mesh we can use a simpler layering
# because there is no topography there.
if not os.path.exists(flat_mesh_path):
    print("Flat mesh does not exist, creating it.")
    mesh_flat = mesh_from_domain(
        domain=d,
        model=lm.MeshingProtocol(
            layered_model_flat,
            interlayer_coarsening_policy=
                layer_coarsening_policy,
            ab=sn.AbsorbingBoundaryParameters(
                reference_velocity=reference_velocity,
                number_of_wavelengths=number_of_wavelengths,
                reference_frequency=absorbing_reference_frequency,
            ),
        ),
        mesh_resolution=mr
    )
    mesh_flat.write_h5(flat_mesh_path)
else:
    print("Flat mesh already exists, loading it.")
    mesh_flat=UnstructuredMesh.from_h5(flat_mesh_path)

tend = time()
# Print the execution time in minutes in 3 decimal places
print(f"Flat mesh created in {(tend - tstart) / 60:.3f} minutes.")

# Establishing the mesh with the topography
print("Creating topography mesh.")

# time the execution of the cell
from time import time
tstart = time()

# Check if the topo mesh exists, other wise create it.
if not os.path.exists(topo_mesh_path):
    print("Topo mesh does not exist, creating it.")
    mesh_topo = mesh_from_domain(
        domain=d,
        model=lm.MeshingProtocol(
            layered_model_topo,
            interlayer_coarsening_policy=
                layer_coarsening_policy,
            ab=sn.AbsorbingBoundaryParameters(
                reference_velocity=reference_velocity,
                number_of_wavelengths=number_of_wavelengths,
                reference_frequency=absorbing_reference_frequency,
            ),
        ),
        mesh_resolution=mr
    )
    mesh_topo.write_h5(topo_mesh_path)
else:
    print("Topo mesh already exists, loading it.")
    mesh_topo=UnstructuredMesh.from_h5(topo_mesh_path)

tend = time()
# Print the execution time in minutes in 3 decimal places
print(f"Topo mesh created in {(tend - tstart) / 60:.3f} minutes.")

# Establish the project
p = sn.Project.from_domain(path=project_path, domain=d, load_if_exists=False)

# %% Define the source and receivers
# Step 6: Define the source and receivers
print("Step 6: Define the source and receivers")

source_file=pd.read_csv(fault_csv_path, delimiter=",")
d=source_file['rise_time'].astype(float)
f=d.to_list()
r=source_file['rup_time'].astype(float)
rupt=r.to_list()

sources=[]
# wavelets=sn.simple_config.stf.Ricker(center_frequency=0.2)
wavelets=[]
for i in range(len(f)):
    wave_pt=sn.simple_config.stf.GaussianRate(half_duration_in_seconds=float(f[i])/2, 
                                              time_shift_in_seconds=float(rupt[i]))
    wavelets.append(wave_pt)

for idx,row in source_file.iterrows():
    event_pt=sn.simple_config.source.cartesian.MomentTensorPoint3D(
    x=row.x_rot,
    y=row.y_rot,
    # Conversion to m is done in cell number 5 of usgs code
    z=row.z * -1,
    mxx= row.Mtt,
    myy= row.Mpp,
    mzz= row.Mrr,
    myz= -row.Mrp,
    mxz= row.Mrt,
    mxy= -row.Mtp,
)

    sources.append(event_pt)

# Create an array for the receivers
rec = sn.simple_config.receiver.cartesian.collections.SideSetArrayPoint3D(
    y=np.arange(9230000, 9447000, 10000),
    x=np.arange(500000, 832000, 10000),
    depth_in_meters=0.0,
    fields=["velocity"],
)

#Adding the event to the project.
p += sn.Event(event_name="PNG", sources=sources, receivers=rec)

# Define the source time function
ec = sn.EventConfiguration(
    wavelet=wavelets,
    waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
        end_time_in_seconds=end_time_in_seconds
    ),
)

p += sn.UnstructuredMeshSimulationConfiguration(
    name="flat", unstructured_mesh=mesh_flat, event_configuration=ec
)

p += sn.UnstructuredMeshSimulationConfiguration(
    name="topo", unstructured_mesh=mesh_topo, event_configuration=ec
)

#%% Visualize the flat mesh simulation configuration
# Step: Visualize 
# p.viz.nb.simulation_setup("flat",["PNG"])

# # Visualize the topo mesh simulation configuration
# p.viz.nb.simulation_setup("topo",["PNG"])

#%% Launch simulation for flat mesh
print("Step 7: Launch simulation for flat mesh")

print("Launching simulation for flat mesh.")
tstart = time()
p.simulations.launch(
    "flat",
    events=["PNG"],
    site_name=SALVUS_FLOW_SITE_NAME, # Change to 'snel_cpu_slurm' for slurm execution
    wall_time_in_seconds_per_job=wall_time_in_seconds,
    ranks_per_job=ranks,
    verbosity=1,
    extra_output_configuration={
        "surface_data": {
            "sampling_interval_in_time_steps": sampling_interval_in_time_steps,
            "fields": ["velocity", "acceleration"],
            "side_sets": ["z1"],
        },
    },
)
p.simulations.query(block=True)

tend = time()
# Print simulation time in minutes
print(f"Flat mesh simulation completed in {(tend - tstart) / 60:.3f} minutes.")

#%% Launch simulation for topo mesh
print("Launching simulation for topo mesh.")
tstart = time()
p.simulations.launch(
    "topo",
    events=["PNG"],
    site_name=SALVUS_FLOW_SITE_NAME, # Change to 'snel_cpu_slurm' for slurm execution
    wall_time_in_seconds_per_job=wall_time_in_seconds,
    ranks_per_job=ranks,
    verbosity=1,
    extra_output_configuration={
        "surface_data": {
            "sampling_interval_in_time_steps": sampling_interval_in_time_steps,
            "fields": ["velocity", "acceleration"],
            "side_sets": ["z1"],
        },
    },
)
p.simulations.query(block=True)

tend = time()
# Print simulation time in minutes
print(f"Topo mesh simulation completed in {(tend - tstart) / 60:.3f} minutes.")

#%% Plot: Visualize time series
# p.viz.nb.waveforms([ "flat","topo"], receiver_field="velocity")
