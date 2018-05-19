# credits due to 6.832 Spring 2018 course
# with modifications by Hao Shen
# for 6.832 final project

from numpy import sin, cos
import numpy as np

# These are only for plotting
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
import mpl_toolkits.mplot3d.art3d as art3d



from pydrake.all import MathematicalProgram

class DroneSim():

    def __init__(self):
        self.G  = 9.8  # gravitational constant

    def drone_dynamics(self, state, u):
        '''
        Calculates the dynamics, i.e.:
           \dot{state} = f(state,u)

        :param state: numpy array, length 6, comprising state of system:
            [x, y, z, \dot{x}, \dot{y}, \dot{z}]
        :param u: numpy array, length 3, comprising control input for system:
            [\ddot{x}_u, \ddot{y}_u,\ddot{z}_u]   
            Note that this is only the added acceleration, note the total acceleration.

        :return: numpy array, length 6, comprising the time derivative of the system state:
            [\dot{x}, \dot{y}, \dot{z}, \ddot{x}, \ddot{y}, \ddot{z}]
        '''

        drone_position = state[0:3]
        derivs = np.zeros_like(state)
        derivs[0:3] = state[3:6]

        G = self.G;

        # includes drag and gravity forces
        # derivs[3] = u[0] - 0.05*state[3] - 1.0*drone_position[2] #wind
        derivs[3] = u[0] - 0.05*state[3]
        derivs[4] = u[1] - 0.05*state[4]
        derivs[5] = u[2] - 0.05*state[5] - G
        
        return derivs

    def two_norm(self, x):
        '''
        Euclidean norm but with a small slack variable to make it nonzero.
        This helps the nonlinear solver not end up in a position where
        in the dynamics it is dividing by zero.

        :param x: numpy array of any length (we only need it for length 2)
        :return: numpy.float64
        '''
        slack = .001
        return np.sqrt(((x)**2).sum() + slack)

    def simulate_states_over_time(self, state_initial, time_array, input_trajectory):
        '''
        Given an initial state, simulates the state of the system.

        This uses simple Euler integration.  The purpose here of not
        using fancier integration is to provide what will be useful reference for
        a simple direct transcription trajectory optimization implementation.

        The first time of the time_array __is__ the time of the state_initial.

        :param state_initial: numpy array of length 6, see drone_dynamics for documentation
        :param time_array: numpy array of length N+1 (0, ..., N) whose elements are samples in time, i.e.:
            [ t_0,
              ...
              t_N ] 
            Note the times do not have to be evenly spaced
        :param input_trajectory: numpy 2d array of N rows (0, ..., N-1), and 2 columns, corresponding to
            the control inputs at each time, except the last time, i.e.:
            [ [u_0, u_1],
              ...
              [u_{N-1}, u_{N-1}] ]

        :return: numpy 2d array where the rows are samples in time corresponding
            to the time_array, and each row is the state at that time, i.e.:
            [ [x_0, y_0, z_0, \dot{x}_0, \dot{y}_0, \dot{z}_0],
              ...
              [x_N, y_N, z_N, \dot{x}_N, \dot{y}_N], \dot{z}_N] ]
        '''
        states_over_time = np.asarray([state_initial])
        for i in range(1,len(time_array)):
            time_step = time_array[i] - time_array[i-1]
            state_next = states_over_time[-1,:] + time_step*self.drone_dynamics(states_over_time[-1,:], input_trajectory[i-1,:])
            states_over_time = np.vstack((states_over_time, state_next))
        return states_over_time

    def get_next_state(self, state_initial, input_cmd, time_step):
        next_state = state_initial + time_step*self.drone_dynamics(state_initial, input_cmd)
        return next_state        
        
    def plot_trajectory(self, trajectory, end_pos):
        '''
        Given a trajectory, plots this trajectory over time.

        :param: trajectory: the output of simulate_states_over_time, or equivalent
            Note: see simulate_states_over_time for documentation of the shape of the output
        '''
        input_trajectory = np.zeros((trajectory.shape[0],3))
        self.plot_trajectory_with_boosters(trajectory, input_trajectory, end_pos)

    def plot_trajectory_with_boosters(self, trajectory, input_trajectory, end_pos):
        '''
        Given a trajectory and an input_trajectory, plots this trajectory and control inputs over time. As well as the goal position

        :param: trajectory: the output of simulate_states_over_time, or equivalent
            Note: see simulate_states_over_time for documentation of the shape of the output
        :param: input_trajectory: the input to simulate_states_over_time, or equivalent
            Note: see simulate_states_over_time for documentation of the shape of the input_trajectory
        :param: end_pos: goal ending position
        '''
        drone_position_x = trajectory[:,0]
        drone_position_y = trajectory[:,1]
        drone_position_z = trajectory[:,2]
        fig = plt.figure(figsize=(8,9), dpi=100)
        axes = fig.gca(projection='3d')
        axes.set_xlim([-5,5])
        axes.set_ylim([-5,5])
        axes.set_zlim([0,10])
        axes.plot(drone_position_x, drone_position_y, drone_position_z)
        p = Circle(end_pos, 0.1)
        axes.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        ## if we have an input trajectory, plot it
        if len(input_trajectory.nonzero()[0]):
            # the quiver plot works best with not too many arrows
            max_desired_arrows = 40
            num_time_steps = input_trajectory.shape[0]

            if num_time_steps < max_desired_arrows:
                downsample_rate = 1 
            else: 
                downsample_rate = num_time_steps / max_desired_arrows

            drone_position_x = drone_position_x[:-1] # don't need the last state, no control input for it
            drone_position_y = drone_position_y[:-1]
            drone_position_z = drone_position_z[:-1]
            drone_booster_x = input_trajectory[::downsample_rate,0]
            drone_booster_y = input_trajectory[::downsample_rate,1]
            drone_booster_z = input_trajectory[::downsample_rate,2]
            Q = axes.quiver(drone_position_x[::downsample_rate], drone_position_y[::downsample_rate], drone_position_z[::downsample_rate], 
                drone_booster_x, drone_booster_y, drone_booster_z, length=0.1, color="red")

        plt.show()

    def compute_opt_trajectory(self, state_initial, goal_func, verbose = True):
        '''
        nonlinear trajectory optimization to land drone starting at state_initial, on a vehicle target trajectory given by the goal_func

        :return: three return args separated by commas:

            trajectory, input_trajectory, time_array

            trajectory: a 2d array with N rows, and 6 columns. See simulate_states_over_time for more documentation.
            input_trajectory: a 2d array with N-1 row, and 4 columns. See simulate_states_over_time for more documentation.
            time_array: an array with N rows. 
        '''
        # initialize math program
        import time 
        start_time = time.time()
        mp = MathematicalProgram()
        num_time_steps = 40

        booster = mp.NewContinuousVariables(3, "booster_0")
        booster_over_time = booster[np.newaxis,:]
        
        state = mp.NewContinuousVariables(6, "state_0")
        state_over_time = state[np.newaxis,:]

        dt = mp.NewContinuousVariables(1, "dt")


        for k in range(1,num_time_steps-1):
            booster = mp.NewContinuousVariables(3, "booster_%d" % k)
            booster_over_time = np.vstack((booster_over_time, booster))
        for k in range(1,num_time_steps):
            state = mp.NewContinuousVariables(6, "state_%d" % k)
            state_over_time = np.vstack((state_over_time, state))


        goal_state = goal_func(dt[0]*39)

        # calculate states over time
        for i in range(1, num_time_steps):
            sim_next_state = state_over_time[i-1,:] + dt[0]*self.drone_dynamics(state_over_time[i-1,:], booster_over_time[i-1,:])
                
            # add constraints to restrict the next state to the decision variables
            for j in range(6):
                mp.AddConstraint(sim_next_state[j] == state_over_time[i,j])
            
            # don't hit ground
            mp.AddLinearConstraint(state_over_time[i,2] >= 0.05)

            # enforce that we must be thrusting within some constraints
            mp.AddLinearConstraint(booster_over_time[i-1,0] <= 5.0)
            mp.AddLinearConstraint(booster_over_time[i-1,0] >= -5.0)
            mp.AddLinearConstraint(booster_over_time[i-1,1] <= 5.0)
            mp.AddLinearConstraint(booster_over_time[i-1,1] >= -5.0)

            # keep forces in a reasonable position
            mp.AddLinearConstraint(booster_over_time[i-1,2] >= 1.0)
            mp.AddLinearConstraint(booster_over_time[i-1,0] <= booster_over_time[i-1,2])
            mp.AddLinearConstraint(booster_over_time[i-1,0] >= -booster_over_time[i-1,2])
            mp.AddLinearConstraint(booster_over_time[i-1,1] <= booster_over_time[i-1,2])
            mp.AddLinearConstraint(booster_over_time[i-1,1] >= -booster_over_time[i-1,2])
            

        # add constraints on initial state
        for i in range(6):
            mp.AddLinearConstraint(state_over_time[0,i] == state_initial[i])
        
        # add constraints on dt
        mp.AddLinearConstraint(dt[0] >= 0.001)

        # add constraints on end state
        
        # end goal velocity
        mp.AddConstraint(state_over_time[-1,0] <= goal_state[0] + 0.01)
        mp.AddConstraint(state_over_time[-1,0] >= goal_state[0] - 0.01)
        mp.AddConstraint(state_over_time[-1,1] <= goal_state[1] + 0.01)
        mp.AddConstraint(state_over_time[-1,1] >= goal_state[1] - 0.01)
        mp.AddConstraint(state_over_time[-1,2] <= goal_state[2] + 0.01)
        mp.AddConstraint(state_over_time[-1,2] >= goal_state[2] - 0.01)
        mp.AddConstraint(state_over_time[-1,3] <= goal_state[3] + 0.01)
        mp.AddConstraint(state_over_time[-1,3] >= goal_state[3] - 0.01)
        mp.AddConstraint(state_over_time[-1,4] <= goal_state[4] + 0.01)
        mp.AddConstraint(state_over_time[-1,4] >= goal_state[4] - 0.01)
        mp.AddConstraint(state_over_time[-1,5] <= goal_state[5] + 0.01)
        mp.AddConstraint(state_over_time[-1,5] >= goal_state[5] - 0.01)

        # add the cost function
        mp.AddQuadraticCost(0.01 * booster_over_time[:,0].dot(booster_over_time[:,0]))
        mp.AddQuadraticCost(0.01 * booster_over_time[:,1].dot(booster_over_time[:,1]))
        mp.AddQuadraticCost(0.01 * booster_over_time[:,2].dot(booster_over_time[:,2]))

        # add more penalty on dt because minimizing time turns out to be more important
        mp.AddQuadraticCost(10000*dt[0]*dt[0])

        solved = mp.Solve()
        if verbose:
            print solved
        
        # extract
        booster_over_time = mp.GetSolution(booster_over_time)
        output_states = mp.GetSolution(state_over_time)
        dt = mp.GetSolution(dt)

        time_array = np.zeros(40)
        for i in range(40):
            time_array[i] = i*dt
        trajectory = self.simulate_states_over_time(state_initial, time_array, booster_over_time)
        
        durations = time_array[1:len(time_array)] - time_array[0:len(time_array)-1]
        fuel_consumption = (np.sum(booster_over_time[:len(time_array)]**2, axis=1) * durations).sum()

        if verbose:
            print 'expected remaining fuel consumption', fuel_consumption
            print("took %s seconds" % (time.time() - start_time ))
            print ''

        return trajectory, booster_over_time, time_array, fuel_consumption


    def compute_trajectory(self, state_initial, goal_state, flight_time, exact = False, verbose=True):
        '''
        nonlinear trajectory optimization to land drone starting at state_initial, to a goal_state, in a specific flight_time

        :return: three return args separated by commas:

            trajectory, input_trajectory, time_array

            trajectory: a 2d array with N rows, and 6 columns. See simulate_states_over_time for more documentation.
            input_trajectory: a 2d array with N-1 row, and 4 columns. See simulate_states_over_time for more documentation.
            time_array: an array with N rows. 

        '''
        # initialize math program
        import time 
        start_time = time.time()
        mp = MathematicalProgram()
        num_time_steps = int(min(40, flight_time/0.05))
        dt = flight_time/num_time_steps
        time_array = np.arange(0.0, flight_time - 0.00001, dt) # hacky way to ensure it goes down
        num_time_steps = len(time_array) # to ensure these are equal lenghts
        flight_time = dt*num_time_steps

        if verbose:
            print ''
            print 'solving problem with no guess'
            print 'initial state', state_initial
            print 'goal state', goal_state
            print 'flight time', flight_time
            print 'num time steps', num_time_steps
            print 'exact traj', exact
            print 'dt', dt
        

        booster = mp.NewContinuousVariables(3, "booster_0")
        booster_over_time = booster[np.newaxis,:]
        
        state = mp.NewContinuousVariables(6, "state_0")
        state_over_time = state[np.newaxis,:]


        for k in range(1,num_time_steps-1):
            booster = mp.NewContinuousVariables(3, "booster_%d" % k)
            booster_over_time = np.vstack((booster_over_time, booster))
        for k in range(1,num_time_steps):
            state = mp.NewContinuousVariables(6, "state_%d" % k)
            state_over_time = np.vstack((state_over_time, state))

        # calculate states over time
        for i in range(1,len(time_array)):
            time_step = time_array[i] - time_array[i-1]
            sim_next_state = state_over_time[i-1,:] + time_step*self.drone_dynamics(state_over_time[i-1,:], booster_over_time[i-1,:])
                
            # add constraints to restrict the next state to the decision variables
            for j in range(6):
                mp.AddLinearConstraint(sim_next_state[j] == state_over_time[i,j])
            
            # don't hit ground
            mp.AddLinearConstraint(state_over_time[i,2] >= 0.05)

            # enforce that we must be thrusting within some constraints
            mp.AddLinearConstraint(booster_over_time[i-1,0] <= 5.0)
            mp.AddLinearConstraint(booster_over_time[i-1,0] >= -5.0)
            mp.AddLinearConstraint(booster_over_time[i-1,1] <= 5.0)
            mp.AddLinearConstraint(booster_over_time[i-1,1] >= -5.0)

            # keep forces in a reasonable position
            mp.AddLinearConstraint(booster_over_time[i-1,2] >= 1.0)
            mp.AddLinearConstraint(booster_over_time[i-1,2] <= 15.0)
            mp.AddLinearConstraint(booster_over_time[i-1,0] <= booster_over_time[i-1,2])
            mp.AddLinearConstraint(booster_over_time[i-1,0] >= -booster_over_time[i-1,2])
            mp.AddLinearConstraint(booster_over_time[i-1,1] <= booster_over_time[i-1,2])
            mp.AddLinearConstraint(booster_over_time[i-1,1] >= -booster_over_time[i-1,2])
            

        # add constraints on initial state
        for i in range(6):
            mp.AddLinearConstraint(state_over_time[0,i] == state_initial[i])
        
        # add constraints on end state
        # 100 should be a lower constant...
        mp.AddLinearConstraint(state_over_time[-1,0] <= goal_state[0] + flight_time/100.)
        mp.AddLinearConstraint(state_over_time[-1,0] >= goal_state[0] - flight_time/100.)
        mp.AddLinearConstraint(state_over_time[-1,1] <= goal_state[1] + flight_time/100.)
        mp.AddLinearConstraint(state_over_time[-1,1] >= goal_state[1] - flight_time/100.)
        mp.AddLinearConstraint(state_over_time[-1,2] <= goal_state[2] + flight_time/100.)
        mp.AddLinearConstraint(state_over_time[-1,2] >= goal_state[2] - flight_time/100.)
        mp.AddLinearConstraint(state_over_time[-1,3] <= goal_state[3] + flight_time/100.)
        mp.AddLinearConstraint(state_over_time[-1,3] >= goal_state[3] - flight_time/100.)
        mp.AddLinearConstraint(state_over_time[-1,4] <= goal_state[4] + flight_time/100.)
        mp.AddLinearConstraint(state_over_time[-1,4] >= goal_state[4] - flight_time/100.)
        mp.AddLinearConstraint(state_over_time[-1,5] <= goal_state[5] + flight_time/100.)
        mp.AddLinearConstraint(state_over_time[-1,5] >= goal_state[5] - flight_time/100.)

        # add the cost function
        mp.AddQuadraticCost(10. * booster_over_time[:,0].dot(booster_over_time[:,0]))
        mp.AddQuadraticCost(10. * booster_over_time[:,1].dot(booster_over_time[:,1]))
        mp.AddQuadraticCost(10. * booster_over_time[:,2].dot(booster_over_time[:,2]))

        for i in range(1, len(time_array)-1):
            cost_multiplier = np.exp(4.5*i/(len(time_array)-1))# exp starting at 1 and going to around 90
            # penalize difference_in_state
            dist_to_goal_pos = goal_state[:3] - state_over_time[i-1,:3]
            mp.AddQuadraticCost(cost_multiplier*dist_to_goal_pos.dot(dist_to_goal_pos))
            # penalize difference in velocity
            if exact:
                dist_to_goal_vel = goal_state[-3:] - state_over_time[i-1,-3:]
                mp.AddQuadraticCost(cost_multiplier/2.*dist_to_goal_vel.dot(dist_to_goal_vel))
            else:
                dist_to_goal_vel = goal_state[-3:] - state_over_time[i-1,-3:]
                mp.AddQuadraticCost(cost_multiplier/6.*dist_to_goal_vel.dot(dist_to_goal_vel))

        solved = mp.Solve()
        if verbose:
            print solved
        
        # extract
        booster_over_time = mp.GetSolution(booster_over_time)

        output_states = mp.GetSolution(state_over_time)
                                       
        durations = time_array[1:len(time_array)] - time_array[0:len(time_array)-1]
        fuel_consumption = (np.sum(booster_over_time[:len(time_array)]**2, axis=1) * durations).sum()
        if verbose:
            print 'expected remaining fuel consumption', fuel_consumption
            print("took %s seconds" % (time.time() - start_time ))
            print 'goal state', goal_state
            print 'end state', output_states[-1]
            print ''

        return output_states, booster_over_time, time_array