import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.time import Duration
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from hw6code.KinematicChain     import KinematicChain
from hw5code.TransformHelpers import *

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

jointnames = [
              'j_body_px', 'j_body_py', 'j_body_pz',
              'j_body_rx', 'j_body_ry', 'j_body_rz',
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
        self.leg_names = ["rf", "rm", "rr", "lf", "lm", "lr"]
        self.chains = [KinematicChain(self, 'base_link', 'tip_' + self.leg_names[l], jointnames[0:6] + jointnames[6 + 3*l: 9 + 3*l]) for l in range(6)]
        self.body_chain = KinematicChain(self, 'base_link', 'MP_BODY', jointnames[0:6])
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
        self.q0 = np.zeros(24)
        self.q0[2] += 0.17

        self.qd = self.q0
        self.pd = self.fkin(self.qd)[0] * 0.9
        self.p0 = self.pd
        self.lam = 20
        self.gamma = 0.01

    def fkin(self, qd):
        J = np.zeros((18, 24))
        xp = np.zeros(18)
        for i, c in enumerate(self.chains):
            xplast, xRlast, Jv, Jw = c.fkin(qd[list(range(6) ) + list(range(6 + 3*i, 9 + 3*i)) ])
            J[3*i: 3*i + 3, 0:6] = Jv[:, 0:6]
            J[3 * i: 3 * i + 3, 6 + 3*i: 9 + 3*i] = Jv[:, 6:]
            xp[3*i:3*i+3] = xplast
        return xp, J

    def fkin_with_body(self, qd, pos = True, rot = True):
        J = np.zeros((24, 24))
        xp = np.zeros(24)
        xp[6:], J[6:, :] = self.fkin(qd)
        xp[0:3], xp[3:6], J[0:3, :], J[3:6, :] = self.body_chain.fkin(qd[list(range(6) )])
        return xp, J

    def walk_traj(self, t, period = 2, dist = 0.2, height=0.08, even = True):
        if even:
            if t % period < period/2:
                return t//period * dist, 0, 0, 0
            h = -sin(t /period * 2 * pi)*height
            dh = -2 * pi / period * cos(t/period*2*pi)*height
            return (t//period + (t%period - period/2))*dist, dist/period*2, h, dh
        else:
            if t % period > period/2:
                return (t//period + 1) * dist, 0, 0, 0
            h = sin(t /period * 2 * pi)*height
            dh = 2 * pi / period * cos(t/period*2*pi)*height
            return (t//period + (t%period))*dist, dist/period*2, h, dh

    def gen_traj(self, t):
        positions = []
        vels = []
        for i, l in enumerate(self.leg_names):
            y = 0
            dy = 0
            x, dx, z, dz = self.walk_traj(t+i*0.6)
            p = np.array([x, y,z])
            dp = np.array([dx, dy, dz])
            positions.append(p)
            vels.append(dp)
        p = np.concatenate(positions) + self.p0
        dp = np.concatenate(vels)
        return p, dp

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
        target_p, target_v = self.gen_traj(self.t)
        # Compute the joints.
        qdlast = self.qd
        xp, J = self.fkin(qdlast)
        # xp = np.concatenate((np.array([0.1 * sin(self.t),0.1 *sin(2*self.t + 1),0.1 *sin(self.t+2) + 0.1]), xp))
        # J = np.vstack((np.zeros((3, 24)), J))
        # J[0,0] = 1
        # J[1, 1] = 1
        # J[2, 2] = 1
        error_p = target_p - xp
        xdot = target_v + error_p * self.lam
        Jwinv = np.linalg.inv(J.T @ J + self.gamma ** 2 * np.eye(24)) @ J.T
        qdsec = np.zeros(24)
        qdsec[3:6] = -50*qdlast[3:6]
        qdsec[6:] = -10 * qdlast[6:]
        qdsec[2] = 20
        qddot = Jwinv @ xdot + (np.eye(24) - Jwinv@J )@qdsec
        qd = qdlast + self.dt * qddot
        self.qd = qd
        self.pd = xp
        # Build up a command message and publish.
        cmdmsg = JointState()
        cmdmsg.header.stamp = self.now().to_msg()  # Current time for ROS
        cmdmsg.name = jointnames  # List of names
        cmdmsg.position = qd.flatten().tolist()  # List of positions
        cmdmsg.velocity = qddot.flatten().tolist()  # List of velocities
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