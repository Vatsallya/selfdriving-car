# selfdriving-car
import numpy as np
import matplotlib.pyplot as plt

class SelfDrivingCar:
    def __init__(self, x=0, y=0, yaw=0, speed=0):
        self.x = x      # Car's x-coordinate
        self.y = y      # Car's y-coordinate
        self.yaw = yaw  # Car's orientation (in radians)
        self.speed = speed  # Car's speed

    def move(self, acceleration, steering_angle, dt=0.1):
        """
        Move the car using a simple kinematic model.
        :param acceleration: Acceleration of the car.
        :param steering_angle: Steering angle in radians.
        :param dt: Time step (default is 0.1 seconds).
        """
        self.x += self.speed * np.cos(self.yaw) * dt
        self.y += self.speed * np.sin(self.yaw) * dt
        self.yaw += self.speed / 2.0 * np.tan(steering_angle) * dt  # Assuming wheelbase = 2.0
        self.speed += acceleration * dt

def pid_control(target, current, kp, kd, ki, integral, prev_error):
    """
    Simple PID controller.
    :param target: Desired target value.
    :param current: Current value.
    :param kp: Proportional gain.
    :param kd: Derivative gain.
    :param ki: Integral gain.
    :param integral: Accumulated integral error.
    :param prev_error: Previous error for derivative calculation.
    :return: Control signal and updated integral/prev_error.
    """
    error = target - current
    derivative = error - prev_error
    integral += error
    control = kp * error + kd * derivative + ki * integral
    return control, integral, error

def follow_path(car, path, dt=0.1):
    """
    Simulate the car following a predefined path.
    :param car: SelfDrivingCar instance.
    :param path: List of (x, y) waypoints.
    :param dt: Time step.
    """
    kp, kd, ki = 1.0, 0.1, 0.01  # PID gains
    integral = 0
    prev_error = 0

    x_trajectory = [car.x]
    y_trajectory = [car.y]

    for waypoint in path:
        target_x, target_y = waypoint
        while np.linalg.norm([car.x - target_x, car.y - target_y]) > 0.1:
            # Calculate steering angle using PID for yaw correction
            target_yaw = np.arctan2(target_y - car.y, target_x - car.x)
            steering_angle, integral, prev_error = pid_control(
                target_yaw, car.yaw, kp, kd, ki, integral, prev_error)

            # Move the car
            car.move(acceleration=0.5, steering_angle=steering_angle, dt=dt)

            # Record trajectory
            x_trajectory.append(car.x)
            y_trajectory.append(car.y)

    return x_trajectory, y_trajectory

# Define path as a series of waypoints
path = [(0, 0), (5, 5), (10, 0), (15, -5), (20, 0)]

# Initialize car
car = SelfDrivingCar()

# Simulate car following the path
x_traj, y_traj = follow_path(car, path)

# Plot the results
path_x, path_y = zip(*path)
plt.figure()
plt.plot(path_x, path_y, 'r--', label="Path")
plt.plot(x_traj, y_traj, 'b-', label="Car Trajectory")
plt.scatter(path_x, path_y, c='red', label="Waypoints")
plt.axis("equal")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Self-Driving Car Path Following")
plt.grid()
plt.show()
