import numpy
from NBody import NBodyStart
from ThreeBody import ThreeBodyStart
from TwoBody import TwoBodyStart

# Define masses
m1 = 1.0 #1.1  # Alpha Centauri A
m2 = 1.0 #0.907  # Alpha Centauri B
m3 = 1.0 # Third Star
# Define initial position vectors
r1 = [-0.25, 0, 0]  # m
r2 = [0.25, 0, 0]  # m
r3 = [0, -0.5, 0]  # m
# Convert pos vectors to arrays
r1 = numpy.array(r1, dtype="float64")
r2 = numpy.array(r2, dtype="float64")
r3 = numpy.array(r3, dtype="float64")
# Define initial velocities
v1 = [0.01, 0.01, 0]  # m/s
v2 = [-0.05, 0, -0.1]  # m/s
v3 = [0, -0.01, 0]  # m/s
# Convert velocity vectors to arrays
v1 = numpy.array(v1, dtype="float64")
v2 = numpy.array(v2, dtype="float64")
v3 = numpy.array(v3, dtype="float64")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Starting Program!')
    init_params = numpy.array([r1, r2, r3, v1, v2, v3])  # Initial parameters
    init_params = init_params.flatten()
    mVec = numpy.array([m1, m2, m3]).flatten()
    # TwoBodyStart()  # Two Body Problem
    # ThreeBodyStart(True, False)  # Three Body Problem
    NBodyStart(init_params, mVec)
    print("Exiting Program!")
