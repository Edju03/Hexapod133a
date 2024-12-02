import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.time import Duration
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState

from hw5code.TransformHelpers import *

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

jointnames = ['j_body_rx', 'j_body_ry', 'j_body_rz',
              'j_c1_rf', 'j_thigh_rf', 'j_tibia_rf',
              'j_c1_rm', 'j_thigh_rm', 'j_tibia_rm',
              'j_c1_rr', 'j_thigh_rr', 'j_tibia_rr',
              'j_c1_lf', 'j_thigh_lf', 'j_tibia_lf',
              'j_c1_lm', 'j_thigh_lm', 'j_tibia_lm',
              'j_c1_lr', 'j_thigh_lr', 'j_tibia_lr']


#
#   Demo Node Class
#
class DemoNode(Node):
    # Initialization.
    def __init__(self, name, rate):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Initialize the transform broadcaster
        self.broadcaster = TransformBroadcaster(self)

        # Add a publisher to send the joint commands.
        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while (not self.count_subscribers('/joint_states')):
            pass

        # Set up the timing so (t=0) will occur in the first update
        # cycle (dt) from now.
        self.dt = 1.0 / float(rate)
        self.t = -self.dt
        self.start = self.get_clock().now() + Duration(seconds=self.dt)

        # Create a timer to keep calling update().
        self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, rate))

    # Shutdown.
    def shutdown(self):
        # Destroy the node, including cleaning up the timer.
        self.destroy_node()

    # Return the current time (in ROS format).
    def now(self):
        return self.start + Duration(seconds=self.t)

    # Update - send a new joint command every time step.
    def update(self):
        # To avoid any time jitter enforce a constant time step and
        # integrate to get the current time.
        self.t += self.dt

        #Compute position/orientation of the pelvis (w.r.t. world).
        ppelvis = pxyz(0, 0, 0)
        Rpelvis = Reye()
        Tpelvis = T_from_Rp(Rpelvis, ppelvis)

        # Build up and send the Pelvis w.r.t. World Transform!
        trans = TransformStamped()
        trans.header.stamp = self.now().to_msg()
        trans.header.frame_id = 'world'
        trans.child_frame_id = 'base_link'
        trans.transform = Transform_from_T(Tpelvis)
        self.broadcaster.sendTransform(trans)

        # Compute the joints.
        q = np.zeros(len(jointnames))
        qdot = np.zeros(len(jointnames))
        # Define the movement pattern for the hexapod
        for i in range(7):
            q[3 * i] = pi / 8 * sin(self.t + i * pi / 3)
            qdot[3 * i] = pi / 8 * cos(self.t + i * pi / 3)
            q[3 * i + 1] = pi / 6 * sin(self.t + i * pi / 3)
            qdot[3 * i + 1] = pi / 6 * cos(self.t + i * pi / 3)
            q[3 * i + 2] = pi / 4 * sin(self.t + i * pi / 3)
            qdot[3 * i + 2] = pi / 4 * cos(self.t + i * pi / 3)

        # Build up a command message and publish.
        cmdmsg = JointState()
        cmdmsg.header.stamp = self.now().to_msg()  # Current time for ROS
        cmdmsg.name = jointnames  # List of names
        cmdmsg.position = q.flatten().tolist()  # List of positions
        cmdmsg.velocity = qdot.flatten().tolist()  # List of velocities
        self.pub.publish(cmdmsg)


#
#  Main Code
#
def main(args=None):
    # Initialize ROS and the demo node (100Hz).
    rclpy.init(args=args)
    node = DemoNode('pirouette', 100)

    # Spin, until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()