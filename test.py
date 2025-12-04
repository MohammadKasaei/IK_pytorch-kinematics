import os
import time
import torch
import pybullet as p
import pybullet_data
import pytorch_kinematics as pk
import numpy as np
def main():
    # 1) Start PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane = p.loadURDF("plane.urdf")
    
    # 2) Load KUKA iiwa URDF — use fixed base
    kuka_urdf = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
    robot_id = p.loadURDF(kuka_urdf, useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    print("KUKA iiwa joints:", num_joints)
    
    # 3) Build PyTorch‑Kinematics chain
    # end effector link name for KUKA iiwa (7‑DoF) is typically "lbr_iiwa_link_7"
    with open(kuka_urdf, "rb") as f:
        urdf_str = f.read()
    chain = pk.build_serial_chain_from_urdf(urdf_str, end_link_name="lbr_iiwa_link_7")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chain = chain.to(device=device)
    print("Chain built:", chain)


    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

    # goal equal to current configuration
    lim = torch.tensor(chain.get_joint_limits(), device=device)
    
    for i in range(100):
        # get the current joint states from pybullet
        joint_states = p.getJointStates(robot_id, range(num_joints))
        cur_q = torch.tensor([js[0] for js in joint_states], device=device)
        M = 1
        rot = pk.transform3d.euler_angles_to_matrix(torch.tensor([0.0, torch.pi/2, 0.0], device=device), convention="XYZ")
        goal_in_rob_frame_tf = pk.Transform3d(default_batch_size=1,
                                            rot = rot,
                                            pos = torch.tensor([0.5+0.1*np.cos(2*torch.pi*i/100), 0.0, 0.5+0.1*np.sin(2*torch.pi*i/100)], dtype=torch.float32, device=device),
                                            device=device)

        
        ik = pk.PseudoInverseIK(chain, max_iterations=30, num_retries=10,
                                joint_limits=lim.T,
                                early_stopping_any_converged=True,
                                early_stopping_no_improvement="all",
                                retry_configs=cur_q.reshape(1, -1),
                                # line_search=pk.BacktrackingLineSearch(max_lr=0.2),
                                debug=False,
                                lr=0.2)

        # do IK
        sol = ik.solve(goal_in_rob_frame_tf)
        joints = sol.solutions[0].squeeze(0).cpu().numpy().tolist()

        # Apply joint positions in PyBullet
        for i, joint_angle in enumerate(joints):
            try:
                p.resetJointState(robot_id, i, joint_angle)
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_angle,
                    force=500
                )
            except Exception as e:
                # some joints may be fixed or not controllable — skip
                continue
        p.stepSimulation()

    print("Done. Press ESC in the GUI to close.")
    while True:
        time.sleep(0.1)  # keep alive so you can inspect


if __name__ == "__main__":
    main()