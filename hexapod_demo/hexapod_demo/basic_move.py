'''Hexapod.py

   This implements a hexapod robot with kinematic chains to climb a staircase.

   Node:        /hexapod
   Broadcast:   'base_link' w.r.t. 'world'
   Subscribe:   /joint_states
   Publish:     /visualization_marker_array

'''

import rclpy
import numpy as np

from rclpy.node                 import Node
from rclpy.time                 import Duration
from tf2_ros                    import TransformBroadcaster
from geometry_msgs.msg          import TransformStamped
from sensor_msgs.msg            import JointState
from geometry_msgs.msg          import Point, Vector3
from std_msgs.msg               import ColorRGBA
from visualization_msgs.msg     import Marker
from visualization_msgs.msg     import MarkerArray
from rclpy.qos                  import QoSProfile, DurabilityPolicy

from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *
from hw6code.KinematicChain     import KinematicChain

from math import pi, sin, cos

jointnames = [
    'j_c1_rf', 'j_thigh_rf', 'j_tibia_rf', 
    'j_c1_rm', 'j_thigh_rm', 'j_tibia_rm', 
    'j_c1_rr', 'j_thigh_rr', 'j_tibia_rr',
    'j_c1_lf', 'j_thigh_lf', 'j_tibia_lf',
    'j_c1_lm', 'j_thigh_lm', 'j_tibia_lm',
    'j_c1_lr', 'j_thigh_lr', 'j_tibia_lr'
]

def step(x, y, z, lx, ly, lz):
    # Create a single step marker
    marker = Marker()
    marker.type = Marker.CUBE
    marker.pose.orientation.w = 1.0
    marker.pose.position = Point(x=x, y=y, z=z)
    marker.scale = Vector3(x=lx, y=ly, z=lz)
    marker.color = ColorRGBA(r=0.8, g=0.6, b=1.0, a=1.0)
    marker.frame_locked = True
    return marker

def staircase():
    markers = []
    
    x0, y0, z0 = 0.0, 0.0, 0.0  
    lx, ly, lz = 0.3, 0.5, 0.2
    
    for i in range(10):
        markers.append(step(x0 + i * lx + lx / 2, y0, z0 + i * lz + lz / 2, lx, ly, lz))
    
    return markers

class HexapodNode(Node):
    # Initialization.
    def __init__(self, name, rate):
        # Initialize the node
        super().__init__(name)

        # Initialize the transform broadcaster
        self.broadcaster = TransformBroadcaster(self)

        # Publisher for joint commands
        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # Publisher for visualization markers
        quality = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/visualization_marker_array', quality)

        self.chains = []
        leg_names = ['rf', 'rm', 'rr', 'lf', 'lm', 'lr']

        for name in leg_names:
            leg_joints = [
                f'j_c1_{name}',
                f'j_thigh_{name}',
                f'j_tibia_{name}'
            ]
            chain = KinematicChain(self, 'base_link', f'tibia_{name}', leg_joints)
            self.chains.append(chain)

        self.qd = np.zeros(len(jointnames))
        self.qddot = np.zeros(len(jointnames))

        self.lam = 20  
        self.dt = 1.0 / float(rate)
        self.t = -self.dt
        self.start = self.get_clock().now() + Duration(seconds=self.dt)

        # Create and publish the staircase markers
        markers = staircase()
        timestamp = self.get_clock().now().to_msg()
        for i, marker in enumerate(markers):
            marker.header.stamp = timestamp
            marker.header.frame_id = 'world'
            marker.ns = 'staircase'
            marker.action = Marker.ADD
            marker.id = i
            marker.lifetime = Duration(seconds=0).to_msg()

        self.marker_pub.publish(MarkerArray(markers=markers))

        self.create_timer(self.dt, self.update)
        self.get_logger().info(
            "Running with dt of %f seconds (%fHz)" % (self.dt, rate)
        )

    def shutdown(self):
        self.destroy_node()

    def now(self):
        return self.start + Duration(seconds=self.t)
    
    def trajectory(self, phase, leg_index):
        stride_length = 0.3  
        step_height = 0.2    
        duty_cycle = 0.5  # Time spent per stair

        gait_phase = (phase + leg_index * pi / 3) % (2 * pi) # Just shift
        phase_ratio = gait_phase / (2 * pi) # 0 to 1

        y_positions = [0.2, 0.1, -0.1, -0.2, -0.1, 0.1]
        y = y_positions[leg_index]

        if phase_ratio < duty_cycle: # Leg on ground
            alpha = phase_ratio / duty_cycle # 0 -1
            x = stride_length * (alpha - 0.5)
            z = -0.2 + (step_height * (self.step_index * alpha))
        else: # Leg on air
            alpha = (phase_ratio - duty_cycle) / (1 - duty_cycle)
            x = stride_length * (alpha - 0.5)
            z = -0.2 + step_height * sin(pi * alpha)

        pd_base = np.array([x, y, z])
        vd_base = np.zeros(3)  

        return pd_base, vd_base

    def jacobian_construction(self, pbase, Rbase):
        damping_factor = 1e-6

        self.step_index = int(self.t // (2 * pi))

        for i in range(6):
            idx = slice(i * 3, (i + 1) * 3)
            phase = self.t

            pd_base, vd_base = self.trajectory(phase, i)

            q_current = self.qd[idx]
            p, R, Jv, _ = self.chains[i].fkin(q_current)

            err = pd_base - p

            Jv_damped_pinv = np.linalg.inv(Jv.T @ Jv + damping_factor * np.eye(3)) @ Jv.T
            qdot = Jv_damped_pinv @ (vd_base + self.lam * err)

            self.qddot[idx] = qdot
            self.qd[idx] += qdot * self.dt

    def update(self):
        self.t += self.dt

        forward_speed = 0.05
        pbase = np.array([
            forward_speed * self.t,
            0.0,
            0.2 # add upward movement to make robot go up
        ])
        Rbase = np.eye(3)
        Tbase = T_from_Rp(Rbase, pbase)
        
        trans = TransformStamped()
        trans.header.stamp = self.now().to_msg()
        trans.header.frame_id = 'world'
        trans.child_frame_id = 'base_link'
        trans.transform = Transform_from_T(Tbase)
        self.broadcaster.sendTransform(trans)

        self.jacobian_construction(pbase, Rbase)

        markers = staircase()
        timestamp = self.get_clock().now().to_msg()
        for i, marker in enumerate(markers):
            marker.header.stamp = timestamp
            marker.header.frame_id = 'world'
            marker.ns = 'staircase'
            marker.action = Marker.ADD
            marker.id = i
            marker.lifetime = Duration(seconds=0).to_msg()

        self.marker_pub.publish(MarkerArray(markers=markers))

        cmdmsg = JointState()
        cmdmsg.header.stamp = self.now().to_msg()
        cmdmsg.name = jointnames
        cmdmsg.position = self.qd.tolist()
        cmdmsg.velocity = self.qddot.tolist()
        self.pub.publish(cmdmsg)

#
#  Main Code
#
def main(args=None):
    # Initialize ROS and the demo node
    rclpy.init(args=args)
    node = HexapodNode('hexapod', 100)

    # Spin until interrupted
    rclpy.spin(node)

    # Shutdown the node and ROS
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
