'''hw7p2.py

   This is a placeholder and skeleton code for HW7 Problem 2.

   Insert your HW7 P1 code and edit for this problem.  The below
   repulsion function will be used in P2(d).  The statements
   thereafter show how it will be integrated into your code.

'''



#
#   Repulsion Joint Torques
#
#   This computes the equivalent joint torques that mimic a repulsion
#   force between the forearm and the top edge of the wall.  It uses
#   the kinematic chains to the elbow (4 joints) and wrist (5 joints)
#   to get the forearm segment.
#
def repulsion(q, wristchain, elbowchain):
    # Compute the wrist and elbow points.
    (pwrist, _, Jv, Jw) = wristchain.fkin(q[0:5])  # 5 joints
    (pelbow, _, _, _)   = elbowchain.fkin(q[0:4])  # 4 joints

    # Determine the wall (obstacle) "line"
    pw = np.array([0, 0, 0.3])
    dw = np.array([0, 1, 0])

    # Determine the forearm "line"
    pa = pwrist
    da = pelbow - pwrist

    # Solve for the closest point on the forearm.
    a = (pw - pa) @ np.linalg.pinv(np.vstack((-dw, np.cross(dw, da), da)))
    parm = pa + max(0, min(1, a[2])) * da

    # Solve for the matching wall point.
    pwall = pw + dw * np.inner(dw, parm-pw) / np.inner(dw, dw)

    # Compute the distance and repulsion force
    d = np.linalg.norm(parm-pwall)
    F = (parm-pwall) / d**2

    # Map the repulsion force acting at parm to the equivalent force
    # and torque actiing at the wrist point.
    Fwrist = F
    Twrist = np.cross(parm-pwrist, F)

    # Convert the force/torque to joint torques (J^T).
    tau = np.vstack((Jv, Jw)).T @ np.concatenate((Fwrist, Twrist))

    # Return the 5 joint torques as part of the 7 full joints.
    return np.concatenate((tau, np.zeros(2)))


#
#   To the Trajectory Class, add:
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        ....

        # Set up the intermediate kinematic chain objects.
        self.chain5 = KinematicChain(
            node, 'world', 'link5', self.jointnames()[0:5])
        self.chain4 = KinematicChain(
            node, 'world', 'link4', self.jointnames()[0:4])

        ....
        

    # Evaluation
    def evaluate(self, t, dt):
        ....

        qsdot = c * repulsion(qdlast, self.chain5, self.chain4)

        ....
