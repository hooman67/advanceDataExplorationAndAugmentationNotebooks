from setuptools import setup

setup(
   name='FMDL_Algo',
   version='1.0',
   description='FMDL Algorithm Module',
   author='Hooman Shariati',
   author_email='hooman@motionmetrics.com',
   packages=['fmdlAlgo', 'detectors', 'utils'],  #same as name
   install_requires=[
        "Keras==2.2.4",
        "matplotlib==3.0.2",
        "numpy==1.15.4",
        "opencv-python==3.4.7.28",
        "Pillow==5.4.1",
        "tensorflow==1.12.0"
    ], #external packages as dependencies
)
