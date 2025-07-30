import os
import sys
import time
import numpy as np
from collections import deque
sys.path.append(os.getcwd())
from legged_gym.envs.h1.h1_config import H1Cfg
import torch 
import h1_low_level_control as h1
from phase_clock import PhaseClock
from isaacgym.torch_utils import *

class Sim2Sim:
    """
      1. Reads low-level robot state from DDS
      2. Builds stacked observations for HugWBC
      3. Runs the policy for actions
      4. Maps actions -> PD targets/torques
      5. Publishes joint target torques/angles back over DDS
    """

    def __init__(self, device: str = "cuda", domain_id: int = 1, interface: str = "lo"):
        # Policy
        self.device = device
        self.policy: torch.jit.ScriptModule = torch.jit.load("/home/jj/Documents/HugWBC/logs/h1_interrupt/model.pt").to(device)
        self.policy.eval()

        # DDS control
        h1.ChannelFactoryInitialize(domain_id, interface)
        self.h1_ctrl = h1.LowLevelControl()
        self.h1_ctrl.Init()

        # Configs
        self.cfg = H1Cfg()

        self.command_scales = torch.tensor([
            self.cfg.normalization.obs_scales.lin_vel,
            self.cfg.normalization.obs_scales.lin_vel,
            self.cfg.normalization.obs_scales.ang_vel,
            self.cfg.normalization.obs_scales.gait_freq_cmd,
            self.cfg.normalization.obs_scales.gait_phase_cmd,
            self.cfg.normalization.obs_scales.gait_phase_cmd,
            self.cfg.normalization.obs_scales.footswing_height_cmd,
            1.0, 1.0, 1.0, 1.0
        ], dtype=torch.float, device=self.device)

        self.dof_names = list(self.cfg.init_state.default_joint_angles.keys())
        self.num_dofs = len(self.dof_names)
        self.num_actions = self.cfg.env.num_actions
        self.gravity_vec = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float, device=self.device, requires_grad=False)
        
        # Initialize some tensors
        self.joint_stiffness = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_damping = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.custom_torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_action = torch.zeros(19, dtype=torch.float, device=self.device)

        # Commands
        cmds = (
            0.0,    # Linear velocity X ( m/s )     ( min: -0.6, max: 2.0 )
            0.0,    # Linear velocity Y ( m/s )     ( min: -0.6, max: 0.6 )
            0.0,    # Angular velocity  ( rad/s )   ( min: -1.0, max: 1.0 )
            1.5,    # Gait frequency    ( Hz )      ( min: 1.5, max: 3.5 )
            0.5,    # Phase offset                  ( 0.5 = walking, 0.0 = hopping )
            0.5,    # Stance duration   ( s? )      ( min: ?, max: ? )
            0.2,    # Swing height      ( m? )      ( min: ?, max: ? )
            -0.1,   # Body height       ( m? )      ( min: -0.3, max: 0.0)
            0.1,    # Body pitch        ( rad )     ( min: 0.0, max: 0.4 )      
            0.0,    # Waist rotation    ( rad )     ( min: ?, max: ? )
            1.0     # ?
        )
        self.commands = torch.tensor(cmds, dtype=torch.float, device=self.device)

        # Time
        self.dt = self.cfg.sim.dt #* self.cfg.control.decimation

        # phase‐clock for contact targets
        self.clock_gen = PhaseClock(self.dt, device=self.device)

        # observations buffer
        self.obs_buffer = deque(maxlen=5)
        #self.obs_buffer_initialized = False
        #obs_empty = torch.zeros(76, dtype=torch.float, device=self.device)
        #for _ in range(self.obs_buffer_len):
        #    self.obs_buffer.append(obs_empty)

        self.h1_ctrl.LowCmdWriteZero()
        self.init_joints()

    def init_joints(self):
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.joint_stiffness[i] = self.cfg.control.stiffness[dof_name]
                    self.joint_damping[i] = self.cfg.control.damping[dof_name]
                    self.custom_torque_limits[i] = self.cfg.control.torque_limits[dof_name]
                    found = True
            if not found:
                print(f"dof_name not found: {dof_name}")
                self.joint_stiffness[i] = 0.
                self.joint_damping[i] = 0.
                self.custom_torque_limits[i] = 100.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        # Mapping from descriptive joint names to SDK2 motor indices
        joint_name_to_index = {
            # Right leg
            'right_hip_yaw_joint': 8,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': 1,
            'right_knee_joint': 2,
            'right_ankle_joint': 11,

            #Left leg
            'left_hip_yaw_joint': 7,
            'left_hip_roll_joint': 3,
            'left_hip_pitch_joint': 4,
            'left_knee_joint': 5,
            'left_ankle_joint': 10,

            'torso_joint': 6,  # waist yaw

            # Right arm
            'right_shoulder_pitch_joint': 12,
            'right_shoulder_roll_joint': 13,
            'right_shoulder_yaw_joint': 14,
            'right_elbow_joint': 15,

            # Left arm
            'left_shoulder_pitch_joint': 16,
            'left_shoulder_roll_joint': 17,
            'left_shoulder_yaw_joint': 18,
            'left_elbow_joint': 19
        }

        #angles = np.zeros(20)
        #for joint_name, angle in self.cfg.init_state.default_joint_angles.items():
        #    motor_index = joint_name_to_index[joint_name]
        #    angles[motor_index] = angle
        
        angles = self.default_dof_pos.cpu().numpy()
        angles = np.insert(angles, h1.H1JointIndex.kNotUsedJoint, 0.0)

        stiffness = self.joint_stiffness.cpu().numpy()
        stiffness = np.insert(stiffness, h1.H1JointIndex.kNotUsedJoint, 100.0)

        damping = self.joint_damping.cpu().numpy()
        damping = np.insert(damping, h1.H1JointIndex.kNotUsedJoint, 2.0)

        print(f"{angles = }")
        print(f"{stiffness = }")
        print(f"{damping = }")

        # TEST
        #while True:
        for _ in range(500):
            self.h1_ctrl.LowCmdWriteJointAngles(angles, stiffness, damping)

            low_state = self.h1_ctrl.low_state
            
            for i, motor in enumerate(low_state.motor_state):
                print(f"motor {i} - q: {motor.q}")
            print()
            time.sleep(self.dt)
        #sys.exit()
        # TEST
            
    def build_single_obs(self, low_state: h1.LowState_) -> torch.Tensor:
        """Returns a 76-D tensor for a single timestep."""

        # single observation contains (76):
        # - proprioception (44) = angular velocity (3) + projected gravity (3) + joint positions (19) +  joint velocities (19)
        # - last actions (19)
        # - last commands (11)
        # - clock (2)

        # command contains (12?): # paper says command is of length 12 but actual used obs is 11?
        # - target linear velocity x and y, target angular velocity (3)
        # - behaviour command (9?)

        # behaviour command contains:
        # - qait frequency
        # - maximum foot swing height
        # - body height
        # - body pitch angle
        # - waist rotation

        # - gait control of length 4
        #   - phase offset
        #   - duty cycle
        #   - phase variable 1
        #   - phase variable 2

        # Proprioception
        base_ang_vel = torch.tensor(low_state.imu_state.gyroscope, dtype=torch.float, device=self.device) # QUAT ROTATE?
        base_quat = torch.tensor(low_state.imu_state.quaternion, dtype=torch.float, device=self.device)
        projected_gravity = quat_rotate_inverse(base_quat.unsqueeze(0), self.gravity_vec.unsqueeze(0)).squeeze(0)

        print(f"{base_quat = }")
        print(f"{self.gravity_vec = }")
        print(f"{projected_gravity = }")

        dof_pos_list = [m.q for m in low_state.motor_state]
        dof_pos_list.pop(h1.H1JointIndex.kNotUsedJoint)
        self.dof_pos = torch.tensor(dof_pos_list, dtype=torch.float, device=self.device)

        dof_vel_list = [m.dq for m in low_state.motor_state]
        dof_vel_list.pop(h1.H1JointIndex.kNotUsedJoint)
        self.dof_vel = torch.tensor(dof_vel_list, dtype=torch.float, device=self.device)

        #print("base_ang_vel shape:", base_ang_vel.shape)
        #print("projected_gravity shape:", projected_gravity.shape)
        #print("dof_pos shape:", self.dof_pos.shape)
        #print("default_dof_pos shape:", self.default_dof_pos.shape)
        #print("dof_vel shape:", self.dof_vel.shape)

        proprioception = torch.cat((
            base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,
            projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos,
            self.dof_vel * self.cfg.normalization.obs_scales.dof_vel
        ), dim=-1)

        clock = self.clock_gen.update(self.commands)

        obs = torch.cat((
            proprioception, 
            self.last_action, 
            self.commands * self.command_scales,
            clock
        ), dim=-1)

        return obs
    
    def compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """

        actions_scaled = actions * self.cfg.control.action_scale
        torques = self.joint_stiffness * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.joint_damping * self.dof_vel
        torque_limits = self.custom_torque_limits

        torques *= 0.75

        return torch.clip(torques, -torque_limits, torque_limits)

    def run(self):
        while True:
            start = time.perf_counter()

            # read LowState
            low_state = self.h1_ctrl.low_state
            if low_state is None:
                print("low_state is None")
                time.sleep(0.001)
                continue

            # build & buffer observations
            obs = self.build_single_obs(low_state)
            #print(obs)
            self.obs_buffer.append(obs)

            if len(self.obs_buffer) < 5:
                continue

            # inference
            obs_stack = torch.stack(list(self.obs_buffer), dim=0)
            obs_batch = obs_stack.unsqueeze(0) 
            with torch.inference_mode():
                actions = self.policy.act_inference(obs_batch).squeeze(0)
            self.last_action = actions.clone()
            
            # TODO Upper body disturbances
            
            clip_actions = self.cfg.normalization.clip_actions
            actions_clipped = torch.clip(actions.clone(), -clip_actions, clip_actions).to(self.device)
            torques = self.compute_torques(actions_clipped)
            
            # send LowCmd
            torques = torques.cpu().numpy()
            torques = np.insert(torques, h1.H1JointIndex.kNotUsedJoint, 0.0)

            stiffness = self.joint_stiffness.cpu().numpy()
            stiffness = np.insert(stiffness, h1.H1JointIndex.kNotUsedJoint, 100.0)

            damping = self.joint_damping.cpu().numpy()
            damping = np.insert(damping, h1.H1JointIndex.kNotUsedJoint, 2.0)

            for i in range(len(torques)):
                print(f"motor {i}: {torques[i]}")

            #for _ in range(self.cfg.control.decimation):
            self.h1_ctrl.LowCmdWriteJointTorques(torques, stiffness, damping)
            

            #sys.exit()

            elapsed = time.perf_counter() - start
            if self.dt - elapsed > 0.0:
                #print(f"{self.dt - elapsed = }")
                time.sleep(self.dt - elapsed)

if __name__ == "__main__":
    controller = Sim2Sim(
        device="cuda",
        domain_id=1,
        interface="lo"
    )
    controller.run()