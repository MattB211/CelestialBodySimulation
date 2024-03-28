import scipy as sci
import numpy
import matplotlib.pyplot as plot
from matplotlib import animation
import scipy.integrate

# *** Define Constants ***
# Define universal gravitation constant
G = 6.67408e-11  # N-m2/kg2
# Reference quantities
m_nd = 1.989e+30  # kg #mass of the sun
r_nd = 5.326e+12  # m #distance between stars in Alpha Centauri
v_nd = 30000  # m/s #relative velocity of earth around the sun
t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s #orbital period of Alpha Centauri
# Net constants
K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
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

# *** Derivative Equations ***
def ThreeBodyEquations(w, t, m1, m2, m3):
    r1 = w[:3]
    r2 = w[3:6]
    r3 = w[6:9]
    v1 = w[9:12]
    v2 = w[12:15]
    v3 = w[15:18]
    r12 = numpy.linalg.norm(r2 - r1)
    r13 = numpy.linalg.norm(r3 - r1)
    r23 = numpy.linalg.norm(r3 - r2)

    dv1bydt = K1 * (((m2 * (r2 - r1)) / (r12 ** 3)) + ((m3 * (r3 - r1)) / (r13 ** 3)))
    dv2bydt = K1 * (((m1 * (r1 - r2)) / (r12 ** 3)) + ((m3 * (r3 - r2)) / (r23 ** 3)))
    dv3bydt = K1 * (((m1 * (r1 - r3)) / (r13 ** 3)) + ((m2 * (r2 - r3)) / (r23 ** 3)))
    dr1bydt = K2 * v1
    dr2bydt = K2 * v2
    dr3bydt = K2 * v3
    r12_derivs = numpy.concatenate((dr1bydt, dr2bydt))
    r_derivs = numpy.concatenate((r12_derivs, dr3bydt))
    v12_derivs = numpy.concatenate((dv1bydt, dv2bydt))
    v_derivs = numpy.concatenate((v12_derivs, dv3bydt))
    derivs = numpy.concatenate((r_derivs, v_derivs))
    return derivs

# *** Create Plots/Animations ***
def ThreeBodyStart(sim, save):
    init_params = numpy.array([r1, r2, r3, v1, v2, v3])  # Initial parameters
    init_params = init_params.flatten()  # Flatten to make 1D array

    # Modify time span and number of points to change graph output
    # Orbitals = Length of Run
    orbitals = 10
    # Points = Amount of Data -> Graph Smoothness
    points = 1000
    time_span = numpy.linspace(0, orbitals, points)  # 10 orbital periods and 500 points

    three_body_sol = sci.integrate.odeint(ThreeBodyEquations, init_params, time_span, args=(m1, m2, m3))
    r1_sol = three_body_sol[:, :3]
    r2_sol = three_body_sol[:, 3:6]
    r3_sol = three_body_sol[:, 6:9]
    r_com_sol = (m1 * r1_sol[:] + m2 * r2_sol[:] + m3 * r3_sol[:]) / (m1 + m2 + m3)

    fig = plot.figure(figsize=(10, 10), dpi=100)
    # Create 3D axes
    model = fig.add_subplot(111, projection="3d")

    if sim:
        def animate(f):
            # Animate the plots the orbits
            model.cla()
            model.scatter(r1_sol[f, 0], r1_sol[f, 1], r1_sol[f, 2], color="blue")
            model.scatter(r2_sol[f, 0], r2_sol[f, 1], r2_sol[f, 2], color="red")
            model.scatter(r3_sol[f, 0], r3_sol[f, 1], r3_sol[f, 2], color="green")
            model.scatter(r_com_sol[f, 0], r_com_sol[f, 1], r_com_sol[f, 2], color="purple")

            # Draw the lines of the orbit
            model.plot(r1_sol[:f, 0], r1_sol[:f, 1], r1_sol[:f, 2], color="blue")
            model.plot(r2_sol[:f, 0], r2_sol[:f, 1], r2_sol[:f, 2], color="red")
            model.plot(r3_sol[:f, 0], r3_sol[:f, 1], r3_sol[:f, 2], color="green")
            model.plot(r_com_sol[:f, 0], r_com_sol[:f, 1], r_com_sol[:f, 2], color="purple")

        # Label our Plot
        model.set_xlabel("X-Axis", fontsize=14)
        model.set_ylabel("Y-Axis", fontsize=14)
        model.set_zlabel("Z-Axis", fontsize=14)
        model.set_title("Visualization of a Three-Body System\n", fontsize=14)
        model.legend(loc="upper left", fontsize=14)

        # Interval makes the graph smoother and run faster / slower
        # Does not have to be orbitals, it is there to improve smoothness consistency
        # Frames should not exceed points or else out-of-bounds error
        ani = animation.FuncAnimation(fig, animate, frames=points, interval=orbitals)

        # Save a gif of the animation
        # WARNING: FILE SIZE CAN BECOME LARGE
        if save:
            writer = animation.PillowWriter(fps=10)
            print("Saving Gif")
            ani.save(filename="ThreeBodyTest.gif", writer=writer)
            print("Finished!")

    else:
        # Plot the orbits
        model.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], color="darkblue")
        model.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], color="red")
        model.plot(r3_sol[:, 0], r3_sol[:, 1], r3_sol[:, 2], color="green")
        # Plot Center of mass
        model.plot(r_com_sol[:, 0], r_com_sol[:, 1], r_com_sol[:, 2], color="purple")

        # Plot the final positions of the stars
        model.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2], color="blue", marker="o", s=100,label="Star A")
        model.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2], color="red", marker="o", s=100,label="Star B")
        model.scatter(r3_sol[-1, 0], r3_sol[-1, 1], r3_sol[-1, 2], color="green", marker="o", s=100,label="Star C")
        # Plot final center of mass
        model.scatter(r_com_sol[-1, 0], r_com_sol[-1, 1], r_com_sol[-1, 2], color="purple", marker="o", s=100)

    # Show the Image/Animation
    plot.show()


# Interesting Starts
# Collision and launch
# m1 = 1.0
# m2 = 1.0
# m3 = 1.0
# r1 = [-1.5, 0, 0]  # m
# r2 = [0.5, 0, 0]  # m
# r3 = [0, 1, 0]  # m
# v1 = [0.01, 0.01, 0]  # m/s
# v2 = [-0.05, 0, -0.1]  # m/s
# v3 = [0, -0.01, 0]  # m/s
