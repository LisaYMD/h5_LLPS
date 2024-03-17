from setuptools import setup, find_packages

setup(
    name='readdy_llps',
    version="0.0.1",
    description="small packages for reading LLPS synapse simulation of readdy h5 file",
    author='LisaYMD',
    classifiers=["Development Status :: 1 - Planning"],
    packages=["h5Traj", "h5Checkpoint"],
    license='MIT'
)
