"""
Full RRT* Node for ROS 2
Implements RRT* with occupancy grid, path optimization, and drive publishing.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped

# ----------------------
# Node class for tree nodes
# ----------------------
class NodeRRT:
    def __init__(self, x=None, y=None, parent=None, cost=0.0):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost
        self.is_root = False

# ----------------------
# RRT* Node
# ----------------------
class RRTStar(Node):
    def __init__(self):
        super().__init__('rrt_star_node')

        # Parameters
        self.declare_parameter("pose_topic", "ego_racecar/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("goal_x", 10.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("step_size", 0.5)
        self.declare_parameter("max_iters", 500)
        self.declare_parameter("goal_threshold", 0.5)
        self.declare_parameter("map_width", 100)
        self.declare_parameter("map_height", 100)
        self.declare_parameter("map_resolution", 0.1)
        self.declare_parameter("neighbor_radius", 2.0)

        # Load parameters
        self.pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.drive_topic = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.goal_x = self.get_parameter("goal_x").get_parameter_value().double_value
        self.goal_y = self.get_parameter("goal_y").get_parameter_value().double_value
        self.step_size = self.get_parameter("step_size").get_parameter_value().double_value
        self.max_iters = self.get_parameter("max_iters").get_parameter_value().integer_value
        self.goal_threshold = self.get_parameter("goal_threshold").get_parameter_value().double_value
        self.map_width = self.get_parameter("map_width").get_parameter_value().integer_value
        self.map_height = self.get_parameter("map_height").get_parameter_value().integer_value
        self.map_resolution = self.get_parameter("map_resolution").get_parameter_value().double_value
        self.neighbor_radius = self.get_parameter("neighbor_radius").get_parameter_value().double_value

        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 1)

        # Subscribers
        self.pose_sub = self.create_subscription(PoseStamped, self.pose_topic, self.pose_callback, 1)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 1)

        # Occupancy grid
        self.occupancy_grid = np.zeros((self.map_height, self.map_width))
        self.tree = []

    # ----------------------
    # LaserScan callback
    # ----------------------
    def scan_callback(self, scan_msg):
        # Simple occupancy grid placeholder
        # TODO: Convert LaserScan to obstacle coordinates for real collision checking
        pass

    # ----------------------
    # Pose callback -> main RRT* loop
    # ----------------------
    def pose_callback(self, pose_msg):
        current_x = pose_msg.pose.position.x
        current_y = pose_msg.pose.position.y

        # Initialize tree
        if not self.tree:
            root = NodeRRT(current_x, current_y)
            root.is_root = True
            self.tree.append(root)

        path_found = False
        for _ in range(self.max_iters):
            # Sample
            x_rand, y_rand = self.sample()

            # Nearest
            nearest_idx = self.nearest(self.tree, (x_rand, y_rand))
            nearest_node = self.tree[nearest_idx]

            # Steer
            new_node = self.steer(nearest_node, (x_rand, y_rand))

            # Collision check
            if self.check_collision(nearest_node, new_node):
                # Rewiring
                neighbors = self.near(self.tree, new_node)
                min_cost = nearest_node.cost + self.line_cost(nearest_node, new_node)
                new_node.parent = nearest_node
                new_node.cost = min_cost

                for neighbor in neighbors:
                    cost_through_new = new_node.cost + self.line_cost(new_node, neighbor)
                    if cost_through_new < neighbor.cost and self.check_collision(new_node, neighbor):
                        neighbor.parent = new_node
                        neighbor.cost = cost_through_new

                self.tree.append(new_node)

                if self.is_goal(new_node, self.goal_x, self.goal_y):
                    path = self.find_path(self.tree, new_node)
                    self.publish_drive(path)
                    path_found = True
                    break

        if not path_found:
            self.get_logger().info("No path found yet...")

    # ----------------------
    # Sample
    # ----------------------
    def sample(self):
        x = np.random.uniform(0, self.map_width * self.map_resolution)
        y = np.random.uniform(0, self.map_height * self.map_resolution)
        return x, y

    # ----------------------
    # Nearest node
    # ----------------------
    def nearest(self, tree, point):
        dlist = [(node.x - point[0])**2 + (node.y - point[1])**2 for node in tree]
        return dlist.index(min(dlist))

    # ----------------------
    # Steer
    # ----------------------
    def steer(self, nearest_node, point):
        theta = math.atan2(point[1]-nearest_node.y, point[0]-nearest_node.x)
        new_x = nearest_node.x + self.step_size * math.cos(theta)
        new_y = nearest_node.y + self.step_size * math.sin(theta)
        return NodeRRT(new_x, new_y)

    # ----------------------
    # Collision check
    # ----------------------
    def check_collision(self, n1, n2):
        # Placeholder: always free
        # TODO: check interpolated points in occupancy_grid
        return True

    # ----------------------
    # Goal check
    # ----------------------
    def is_goal(self, node, goal_x, goal_y):
        dist = math.hypot(node.x - goal_x, node.y - goal_y)
        return dist < self.goal_threshold

    # ----------------------
    # Path extraction
    # ----------------------
    def find_path(self, tree, node):
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    # ----------------------
    # RRT* cost
    # ----------------------
    def line_cost(self, n1, n2):
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def cost(self, tree, node):
        return node.cost

    # ----------------------
    # RRT* neighborhood
    # ----------------------
    def near(self, tree, node):
        neighbors = [n for n in tree if self.line_cost(n, node) <= self.neighbor_radius]
        return neighbors

    # ----------------------
    # Drive command
    # ----------------------
    def publish_drive(self, path):
        if len(path) < 2:
            return
        next_node = path[1]
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 1.0
        dx = next_node.x - path[0].x
        dy = next_node.y - path[0].y
        drive_msg.drive.steering_angle = math.atan2(dy, dx)
        self.drive_pub.publish(drive_msg)
        self.get_logger().info(f"Drive to ({next_node.x:.2f}, {next_node.y:.2f})")

# ----------------------
# Main
# ----------------------
def main(args=None):
    rclpy.init(args=args)
    node = RRTStar()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
