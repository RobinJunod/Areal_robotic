# You can change anything in this file except the file name of 'my_control.py',
# the class name of 'MyController', and the method name of 'step_control'.

# Available sensor data includes data['t'], data['x_global'], data['y_global'],
# data['roll'], data['pitch'], data['yaw'], data['v_forward'], data['v_left'],
# data['range_front'], data['range_left'], data['range_back'],
# data['range_right'], data['range_down'], data['yaw_rate'].

import numpy as np
import math
import matplotlib as plt
# Don't change the class name of 'MyController'
class MyController():
    def __init__(self):
        self.on_ground = True
        self.rotate = True
        # Default parameters
        self.height_desired = 0.5
        self.max_speed = 0.3
        self.rotation_speed = 0.6
        # Bools for obstacle avoidance
        self.is_avoiding = False
        self.avoid_right = False
        # Path planning
        self.index_current_setpoint = 0
        self.setpoints = [[3.5, 0.0], [3.5, 3.0], [4.0, 3.0], [4.0, 0.0], [4.5, 0.0],  [4.5, 3.0], [5.0, 3.0], [5.0, 0.0]]
        # Starting and Landing pad
        self.go_start_pad = False
        self.start_pad_coord = []
        self.pad_found = False
        
    # Obstacle avoidance with range sensors
    def obstacle_avoidance(self, sensor_data, goal_relative_direction):
        
        # make a simple avoidance algo (potential)
        front = sensor_data['range_front']
        back = sensor_data['range_back']
        right = sensor_data['range_right']
        left = sensor_data['range_left']
        # Compute desired direction and init
        desiered_dir = np.array([goal_relative_direction[0], goal_relative_direction[1]])
        desiered_dir_normal = np.array([desiered_dir[1], -desiered_dir[0]])
        direction = desiered_dir # default direction
        
        if front<0.3 or back<0.3 or right<0.3 or left<0.3:

            # Compute repultion vector
            repulse_front = np.array([-0.2/(0.1+front), 0])
            repulse_back = np.array([0.2/(0.1+back), 0])
            repulse_right = np.array([0, 0.2/(0.1+right)])
            repulse_left = np.array([0, -0.2/(0.1+left)])
            repulse_dir = repulse_back + repulse_front + repulse_left + repulse_right
            
            if self.is_avoiding:
                if self.avoid_right:
                    # avoid obstacle left
                    direction = np.dot(np.array([[0, -1], [1, 0]]), repulse_dir)
                else:
                    # avoid obstacle right
                    direction = np.dot(np.array([[0, 1], [-1, 0]]), repulse_dir)
            
            # Initialize the avoidance direction
            if (front<0.2 or back<0.2 or right<0.2 or left<0.2) and self.is_avoiding==False:
            
                if desiered_dir.dot(repulse_dir)>0 and self.is_avoiding==False: # Don't act if the obstacle is not on the way
                    direction = desiered_dir
                else:
                    if desiered_dir_normal.dot(repulse_dir) > 0:
                        direction = np.dot(np.array([[0, -1], [1, 0]]), repulse_dir) # -90 deg in repulse dir
                        #direction = np.dot(np.array([[0, 1], [-1, 0]]), desiered_dir)
                        self.is_avoiding=True
                        self.avoid_right=True
                        print('avoid right')
                    else:
                        direction = np.dot(np.array([[0, 1], [-1, 0]]), repulse_dir) # +90 deg in repulse dir
                        #direction = np.dot(np.array([[0, -1], [1, 0]]), desiered_dir)
                        self.is_avoiding=True
                        self.avoid_right=False
                        print('avoid left')
        
        else:
            self.is_avoiding=False
            
        direction = self.normalize_speed(direction)
        return [direction[0], direction[1], 0.0, self.height_desired]       
    
    
    
    def get_obj_dir(self, sensor_data):
        """
        Compute the direction vector to the next objective in the global reference frame
        Args:
            sensor_data (list): sensors values

        Returns:
            control_command (numpy vector) : [speed in x, speed in y, rotation speed, height of the drone]
        """
        objective_direction = np.array([0,0])

        # Hover at the final setpoint
        if self.index_current_setpoint == len(self.setpoints):
            objective_direction = np.array([0,0])
            return objective_direction

        # Get the goal position and drone position
        x_goal, y_goal = self.setpoints[self.index_current_setpoint]
        goal_pos = [x_goal, y_goal]
        x_drone, y_drone = sensor_data['x_global'], sensor_data['y_global']
        
        distance_drone_to_goal = np.linalg.norm([x_goal - x_drone, y_goal- y_drone])

        # When the drone reaches the goal setpoint, e.g., distance < 0.1m
        if distance_drone_to_goal < 0.1:
            # Select the next setpoint as the goal position
            self.index_current_setpoint += 1
            self.rotate=True
            # Hover at the final setpoint
            if self.index_current_setpoint == len(self.setpoints):
                objective_direction = np.array([0,0])
                return goal_pos, objective_direction

        # Calculate the control command based on current goal setpoint
        x_goal, y_goal = self.setpoints[self.index_current_setpoint]
        x_drone, y_drone = sensor_data['x_global'], sensor_data['y_global']
        d_x, d_y = x_goal - x_drone, y_goal - y_drone
        objective_direction = np.array([d_x,d_y])
        
        return goal_pos, objective_direction
        

    def compute_roation_speed(self, sensor_data, goal_x, goal_y):
        yaw = sensor_data['yaw']
        #yaw_rate = sensor_data['yaw_rate']
        x_pos, y_pos = sensor_data['x_global'], sensor_data['y_global']
        angle = np.arctan2(goal_y-y_pos, goal_x-x_pos)
        if (angle - yaw) < 0:
            yaw_speed = - self.rotation_speed
        else:
            yaw_speed = self.rotation_speed
        if abs(angle-yaw)<0.01:
            self.rotate = False
        return yaw_speed

    def normalize_speed(self, vector):
        norm = np.linalg.norm(vector)
        if norm < self.max_speed:
            return vector
        return vector / norm * self.max_speed 
    
    def globalCoord_to_droneCoord(self, sensor_data, global_coord_x, global_coord_y):
        # Return a np array of the coord in drone basis
        yaw =  sensor_data['yaw']
        global_coord = np.array([global_coord_x, global_coord_y])
        rot_matrix = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        return rot_matrix.dot(global_coord)
    
    # Don't change the method name of 'step_control'
    def step_control(self, sensor_data):
        # Take off
        if self.on_ground and sensor_data['range_down'] < 0.49 and (not self.pad_found):
            if not (self.pad_found or len(self.start_pad_coord)):
                self.start_pad_coord = [sensor_data['x_global'], sensor_data['y_global']]
                print(self.start_pad_coord)
            control_command = [0.0, 0.0, 0.0, self.height_desired]
            return control_command
        
        # Checking for the landing pad
        elif (not self.on_ground) and sensor_data['range_down'] < 0.45 \
        and sensor_data['range_front']>0.3 and sensor_data['range_left']>0.3 \
        and sensor_data['range_back']>0.3 and sensor_data['range_right']>0.3 and (not self.pad_found):
            self.pad_found = True
            control_command = [0.0, 0.0, 0.0, self.height_desired]
            return control_command
        
        # Land on landing pad
        elif self.pad_found and (not self.go_start_pad):
            control_command = [0.0, 0.0, 0.0, sensor_data['range_down']/2]
            if sensor_data['range_down'] < 0.015:
                print('caraambaaaa pourquoi il ')
                self.go_start_pad = True
                self.on_ground = True
                # add the starting pad coord as next objective
                self.setpoints.append(self.start_pad_coord)
            return control_command
        
        # Take off for the starting pad
        elif self.pad_found and self.go_start_pad and sensor_data['range_down'] < 0.49 and self.on_ground:
            print('ici cest gooood')
            control_command = [0.0, 0.0, 0.0, self.height_desired]
            return control_command            
        
        else:
            self.on_ground = False
            goal_xy, goal_direction = self.get_obj_dir(sensor_data)
            goal_rel_dir = self.globalCoord_to_droneCoord(sensor_data, goal_direction[0], goal_direction[1])
            
            if self.rotate==True:
                rot_speed = self.compute_roation_speed(sensor_data, goal_xy[0], goal_xy[1])
                control_command = [0.0, 0.0, rot_speed, self.height_desired]
            else:
                control_command = self.obstacle_avoidance(sensor_data, goal_rel_dir)
            
                
            return control_command
    




class Occupancy_map():
    def __init__(self):
        # Occupancy map based on distance sensor
        min_x, max_x = 0, 5.0 # meter
        min_y, max_y = 0, 5.0 # meter
        range_max = 2.0 # meter, maximum range of distance sensor
        res_pos = 0.2 # meter
        conf = 0.2 # certainty given by each measurement
        self.map = np.zeros((int((max_x-min_x)/res_pos), int((max_y-min_y)/res_pos))) # 0 = unknown, 1 = free, -1 = occupied

    def update(self, sensor_data):
        pos_x = sensor_data['x_global']
        pos_y = sensor_data['y_global']
        yaw = sensor_data['yaw']

        for j in range(4): # 4 sensors
            yaw_sensor = yaw + j*np.pi/2 #yaw positive is counter clockwise
            if j == 0:
                measurement = sensor_data['range_front']
            elif j == 1:
                measurement = sensor_data['range_left']
            elif j == 2:
                measurement = sensor_data['range_back']
            elif j == 3:
                measurement = sensor_data['range_right']
            
            for i in range(int(Occupancy_map.range_max/Occupancy_map.res_pos)): # range is 2 meters
                dist = i*Occupancy_map.res_pos
                idx_x = int(np.round((pos_x - Occupancy_map.min_x + dist*np.cos(yaw_sensor))/Occupancy_map.res_pos,0))
                idx_y = int(np.round((pos_y - Occupancy_map.min_y + dist*np.sin(yaw_sensor))/Occupancy_map.res_pos,0))

                # make sure the point is within the map
                if idx_x < 0 or idx_x >= map.shape[0] or idx_y < 0 or idx_y >= map.shape[1] or dist > Occupancy_map.range_max:
                    break

                # update the map
                if dist < measurement:
                    map[idx_x, idx_y] += Occupancy_map.conf
                else:
                    map[idx_x, idx_y] -= Occupancy_map.conf
                    break

        map = np.clip(map, -1, 1) # certainty can never be more than 100%
        return 0
    
    def plot(self, t=0): 
        # only plot every Nth time step (comment out if not needed)
        if t % 50 == 0:
            plt.imshow(np.flip(map,1), vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
            plt.savefig("map.png")
        t +=1
        return map