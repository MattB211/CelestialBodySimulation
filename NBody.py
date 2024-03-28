import numpy
import matplotlib.pyplot as plot
from matplotlib import animation
import scipy.integrate

# *** Define Constants ***
# Define universal gravitation constant in k
grav = 6.67430e-20  # kN-km2/kg2
# Reference quantities
m_nd = 1.989e+30  # kg #mass of the sun
r_nd = 5.326e+12  # m #distance between stars in Alpha Centauri
v_nd = 30000  # m/s #relative velocity of earth around the sun
t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s #orbital period of Alpha Centauri
# Net constants
K1 = grav * t_nd * m_nd / (r_nd ** 2 * v_nd)
K2 = v_nd * t_nd / r_nd

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
# Find Centre of Mass
r_com = (m1*r1+m2*r2+m3*r3)/(m1+m2+m3)
# Define initial velocities
v1 = [0.01, 0.01, 0]  # m/s
v2 = [-0.05, 0, -0.1]  # m/s
v3 = [0, -0.01, 0]  # m/s
# Convert velocity vectors to arrays
v1 = numpy.array(v1, dtype="float64")
v2 = numpy.array(v2, dtype="float64")
v3 = numpy.array(v3, dtype="float64")
# Find velocity of COM
v_com = (m1 * v1 + m2 * v2 + m3 * v3) / (m1 + m2 + m3)

def CalcConstants(sVec, mvec):
    numBodies = int(len(sVec)/6)
    # Do we need transpose
    position = numpy.reshape(sVec[:(3 * numBodies)], (3, numBodies))

    # Mass Matrix
    gMass = grav * mvec * numpy.transpose(mvec)

    # Vector to calculate dX, dY, and dZ
    dX = numpy.subtract.outer(numpy.transpose(position[:, 0]), position[:, 0])
    dY = numpy.subtract.outer(numpy.transpose(position[:, 1]), position[:, 1])
    dZ = numpy.subtract.outer(numpy.transpose(position[:, 2]), position[:, 2])

    # Total magnitude of distance
    magDist = numpy.sqrt(numpy.power(dX, 2) + numpy.power(dY, 2) + numpy.power(dZ, 2))
    numpy.fill_diagonal(magDist, 1)

    # Calculate unit normals of the forces
    cX = dX / magDist
    cY = dY / magDist
    cZ = dZ / magDist

    # Calculate the force on each body
    fX = gMass / numpy.power(magDist, 2) * cX
    fY = gMass / numpy.power(magDist, 2) * cY
    fZ = gMass / numpy.power(magDist, 2) * cZ

    # Calculate acceleration
    aX = fX / mvec
    aY = fY / mvec
    aZ = fZ / mvec

    numpy.fill_diagonal(aX, 0)
    numpy.fill_diagonal(aY, 0)
    numpy.fill_diagonal(aZ, 0)

    aX = numpy.sum(aX, axis=1)
    aY = numpy.sum(aY, axis=1)
    aZ = numpy.sum(aZ, axis=1)

    dXdt = numpy.array((numpy.concatenate((sVec[3 * numBodies:],)), numpy.reshape(numpy.transpose([aX, aY, aZ]), (numBodies * 3, 1))))

    return dXdt

def NBodyStart(init_params, mVec):
    numBodies = len(init_params)

    # Integrate over 2 Years
    tSpan = numpy.linspace(0, 365 * 24 * 60 * 60, int(365 * 24 * 60 * 60 / 3600))

    # Perform integration
    dXdT = CalcConstants(init_params, mVec)
    nbody_obj = scipy.integrate.RK45(dXdT, tSpan[0], init_params, tSpan[-1])

    bodies = []
    for k in range(1, numBodies + 1, 1):
        bodies[k] = nbody_obj[:, :(3 * k)]

    fig = plot.figure(figsize=(10, 10), dpi=100)
    # Create 3D axes
    model = fig.add_subplot(111, projection="3d")
    for body in bodies:
        bColor = numpy.random.choice(range(256), size=3)
        # Plot the orbits
        model.plot(body[:, 0], body[:, 1], body[:, 2], color=bColor)
        # Plot the final positions of the stars
        model.scatter(body[-1, 0], body[-1, 1], body[-1, 2], color=bColor, marker="o", s=100)

    # r_com_sol = (m1 * r1_sol[:] + m2 * r2_sol[:] + m3 * r3_sol[:]) / (m1 + m2 + m3)
    # Plot Center of mass
    # model.plot(r_com_sol[:, 0], r_com_sol[:, 1], r_com_sol[:, 2], color="purple")

    # Plot final center of mass
    # model.scatter(r_com_sol[-1, 0], r_com_sol[-1, 1], r_com_sol[-1, 2], color="purple", marker="o", s=100)

    # Show the Image/Animation
    plot.show()
