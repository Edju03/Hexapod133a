import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.time import Duration
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Point, Vector3
from sensor_msgs.msg import JointState
from hw6code.KinematicChain import KinematicChain
from hw5code.TransformHelpers import *
from hw5code.TrajectoryUtils import *
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSProfile, DurabilityPolicy

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
    
    x0, y0, z0 = 0.28, 0.0, 0.0  
    lx, ly, lz = 0.5, 0.8, 0.06
    
    for i in range(10):
        markers.append(step(x0 + i * lx + lx / 2, y0, z0 + i * lz + lz / 2, lx, ly, lz))
    
    return markers


class LegTrajectory:
    def __init__(self, init_pos, leg_num, step_dist = np.array([0.25, 0, 0.06]), step_dis_second = [0.1, 0, 0]):
        self.step_time = 0.4
        self.last = init_pos
        self.target = init_pos
        self.start_time = 0
        self.moving = False
        self.height = 0.6
        self.cur_p = self.last
        self.period = 2
        self.default_step = step_dist
        self.leg_num = leg_num
        self.step_dis_second = step_dis_second
        self.caterpillar = 0
        self.step_dis_third = self.default_step - self.step_dis_second
        self.step_dis_complement = np.array([0.5, 0, 0.06]) - self.default_step
        self.step_height = 0.06

    def get_traj(self, t):
        if t > self.period + self.start_time:
            if self.leg_num in [2, 5]:
                if self.caterpillar == 0:
                    self.set_target(t, self.last + self.step_dis_second)
                elif self.caterpillar == 1:
                    self.set_target(t, self.last + self.step_dis_third - np.array([0, 0, self.step_height]))
                else:
                    self.set_target(t, self.last + self.step_dis_complement + np.array([0, 0, self.step_height]))
                self.caterpillar = (self.caterpillar + 1) % 3
            elif self.leg_num in [1, 4]:
                if self.caterpillar == 0:
                    self.set_target(t, self.last + self.step_dis_second)
                elif self.caterpillar == 1:
                    self.set_target(t, self.last + self.step_dis_third - np.array([0, 0, self.step_height]))
                else:
                    self.set_target(t, self.last + self.step_dis_complement + np.array([0, 0, self.step_height]))
                self.caterpillar = (self.caterpillar + 1) % 3
            else:
                if self.caterpillar == 0:
                    self.set_target(t, self.last + self.default_step)
                elif self.caterpillar == 1:
                    self.set_target(t, self.last + self.step_dis_complement)
                else:
                    self.set_target(t, self.last)
                self.caterpillar = (self.caterpillar + 1) % 3
                
        if t > self.step_time + self.start_time:
            self.moving = False
            self.last = self.target
        if not self.moving:
            return self.last, np.zeros(3)
        v0 = np.array([0, 0, self.height])
        vf =  np.array([0, 0, -self.height])
        a, da = goto(t - self.start_time, self.step_time, 0, self.step_time)

        p, v = spline(a,self.step_time, self.last, self.target, v0, vf)
        v = v * da
        self.cur_p = p
        return p, v

    def set_target(self, t, target):
        self.target = target
        self.last = self.cur_p
        self.start_time = t
        self.moving = True

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
        self.chain_to_body = KinematicChain(self, 'base_link', 'MP_BODY', jointnames[:6])
        # Initialize the transform broadcaster
        self.broadcaster = TransformBroadcaster(self)

        # Add a publisher to send the joint commands.
        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # Add publisher for visualization markers
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', quality)

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
        self.pd = self.fkin(self.qd)[0]
        self.leg_trajs = [LegTrajectory(self.pd[3*l:3*l+3]* np.array([1, 1.2, 1]), l) for l in range(6)]
        for i in range(6):
            pair_number = abs(i % 3  - 2)
            self.leg_trajs[i].start_time = (0.4 * pair_number) % 2

        self.pbody = self.fkin(self.qd, pbody = True)[3]
        self.Rbody = self.fkin(self.qd, Rbody = True)[2]
        self.p0 = self.pd
        self.lam = 20
        self.gamma = 0.01
        self.stair_height = 0.06
        self.stair_width = 0.2

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

    def fkin(self, qd, pbody = False, Rbody = False):
        J = np.zeros((18, 24))
        xp = np.zeros(18)
        for i, c in enumerate(self.chains):
            xplast, xRlast, Jv, Jw = c.fkin(qd[list(range(6) ) + list(range(6 + 3*i, 9 + 3*i)) ])
            J[3*i: 3*i + 3, 0:6] = Jv[:, 0:6]
            J[3 * i: 3 * i + 3, 6 + 3*i: 9 + 3*i] = Jv[:, 6:]
            # Zero out Jacobian for middle legs (indices 1 and 4)
            if i == 2 or i == 4 or i == 0:
                J[3*i: 3*i + 3, :] = 0
            xp[3*i:3*i+3] = xplast
        
        Rlast = None
        plast = None
        if Rbody:
            _, Rlast, _, Jws = self.fkin_s(qd)
            Jw = np.zeros((3, 24))
            Jw[:,:6] = Jws
            J = np.vstack((Jw, J))
        
        if pbody:
            plast, _, Jvs, _ = self.fkin_s(qd)
            Jv = np.zeros((3, 24))
            Jv[:,:6] = Jvs
            J = np.vstack((Jv, J))

        return xp, J, Rlast, plast
    
    # Function to calculate the "inverse" of singular value matrix
    def sinv(self, s, jacobian_shape):
        
        m ,n = jacobian_shape[0], jacobian_shape[1]

        for i in range(len(s)):

            if abs(s[i]) >= self.gamma:
                s[i] = 1/s[i]
            else:
                s[i] = s[i]/(self.gamma)**2

        s = np.diag(s)
        if m < n:
            s = np.hstack((s, np.zeros((m, n-m))))
        elif n < m:
            s = np.vstack((s, np.zeros((m-n, n))))

        return s.T
    
    def fkin_s(self, qd):
        xplast, xRlast, Jv, Jw = self.chain_to_body.fkin(qd[:6])
        return xplast, xRlast, Jv, Jw

    def step_cycle(self, t):
        positions = []
        vels = []

        for i, l in enumerate(self.leg_names):
            trajectory = self.leg_trajs[i]

            cp, cv = trajectory.get_traj(t)
            positions.append(cp)
            vels.append(cv)
        p = np.concatenate(positions)
        dp = np.concatenate(vels)
        return p, dp
    # Shutdown.
    def shutdown(self):
        # Destroy the node, including cleaning up the timer.
        self.destroy_node()

    # Return the current time (in ROS format).
    def now(self):
        return self.start + Duration(seconds=self.t)
    
    def vr_body(self, target_v):
        t = self.t
        period = 5
        phase = t % period
        
        # Initialize lists to store stance leg indices
        stance_legs = []
        
        # Check which legs are in stance phase
        for i in range(6):
            _, _, z, _ = self.walk_traj(t, i)
            if abs(z) < 0.001:  # If leg is not lifting (approximately at ground level)
                stance_legs.append(i)
        
        # Calculate average position using only stance legs
        if stance_legs:  # If we have any stance legs
            pbody_last = np.array([
                np.average([self.pd[3*i] for i in stance_legs]),
                np.average([self.pd[3*i + 1] for i in stance_legs]),
                np.average([self.pd[3*i + 2] for i in stance_legs])
            ])
            pbody_last[2] += 0.17  # Maintain constant height above stance feet
        else:
            pbody_last = self.pbody
        
        # Set desired body velocity (forward motion)
        target_vbody = np.array([0.04, 0, self.stair_height/5])
        
        error_pbody = ep(pbody_last, self.pbody)
        vr_body = target_vbody + error_pbody * self.lam

        return vr_body

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

        # Update and publish staircase markers
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

        target_p, target_v = self.step_cycle(self.t)
        qdlast = self.qd

        xp, J, Rlast, plast = self.fkin(qdlast, pbody = False,  Rbody = True)

        u, s, vT = np.linalg.svd(J)

        Jwinv = vT.T @ (self.sinv(s, J.shape) @ u.T)

        #Jwinv = np.linalg.inv(J.T @ J + self.gamma ** 2 * np.eye(J.T.shape[0])) @ J.T
        error_p = target_p - xp
        xdot = target_v + error_p * self.lam


        if plast is None and Rlast is not None:
            error_R = eR(Reye(), self.Rbody)
            wr = error_R * self.lam
            xdot = np.concatenate((wr, xdot))
            self.Rbody = Rlast
        elif plast is not None and Rlast is None:
            vr_body = self.vr_body(target_v)
            xdot = np.concatenate((vr_body, xdot))
            self.pbody = plast
        elif plast is not None and Rlast is not None:
            error_R = eR(Reye(), self.Rbody)
            wr = error_R * self.lam
            vr_body = self.vr_body(target_v)
            xdot = np.concatenate((vr_body, wr, xdot))
            self.Rbody = Rlast
            self.pbody = plast

        qdsec = np.zeros(J.T.shape[0])
        qdsec[2] = 100
        qdsec[6:] = -20*self.qd[6:]
        qddot = Jwinv @ xdot + (np.eye(J.T.shape[0]) - Jwinv@J )@qdsec
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
