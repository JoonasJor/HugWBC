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

class Sim2Sim:
    """
      1. Reads low-level robot state from DDS
      2. Builds stacked observations for HugWBC
      3. Runs the policy for actions
      4. Maps actions -> PD targets/torques
      5. Publishes joint target torques/angles back over DDS
    """

    def __init__(self, device: str = "cuda", domain_id: int = 1, interface: str = "lo"):
        self.device = device
        # --- Policy & mapping fn ---
        self.policy: torch.jit.ScriptModule = torch.jit.load("/home/jj/Documents/HugWBC/logs/h1_interrupt/model.pt").to(device)
        self.policy.eval()

        # --- DDS control ---
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
            0.0, 0.0, 0.0, 0.0
        ], dtype=torch.float, device=self.device)

        self.dof_names = list(self.cfg.init_state.default_joint_angles.keys())
        self.num_dofs = len(self.dof_names)
        self.num_actions = self.cfg.env.num_actions
        
        """default_joint_angles = [
            0.00,   0.02,   -0.4,   0.8,    -0.4,   # Left leg
           -0.00,  -0.02,   -0.4,   0.8,    -0.4,   # Right Leg
            0.0,                                    # Torso
            0.0,    0.0,    0.0,    0.0,            # Left arm
            0.0,    0.0,    0.0,    0.0             # Right arm
        ]"""
        
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.custom_torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        # Commands
        cmds = (
            2.0,    # linear velocity x
            0.0,    # linear velocity y
            0.0,    # angular velocity
            2.0,    # gait frequency
            0.5,    # gait phase offset
            0.5,    # stance duration
            0.2,    # swing height
            0.0,    # ?
            0.0,    # body pitch
            0.0,    # waist rotation
            0.0     # ?
        )
        self.commands = torch.tensor(cmds, dtype=torch.float, device=self.device)
        #self.use_disturb            = False
        #self.disturb_mask           = torch.ones(1,  self.env_action_dim, dtype=torch.bool, device=self.device)
        #self.disturb_isnoise        = torch.tensor([True], device=self.device)
        #self.disturb_rad_curriculum = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        #self.interrupt_mask         = self.disturb_mask.clone()
        #self.standing_mask          = torch.ones(1, dtype=torch.bool, device=self.device)  # ei näin. selvitä miten toimii oikeasti
        # last action
        self.last_action = torch.zeros(19, dtype=torch.float, device=self.device)

        # Time
        self.dt = self.cfg.sim.dt
        self.t = 0.0

        # phase‐clock for contact targets
        self.clock_gen = PhaseClock(self.dt, device=self.device)

        # history buffer of last `buffer_len` obs
        buffer_len = 5
        self.obs_buffer = deque(maxlen=buffer_len)
        obs_empty = torch.zeros(76, dtype=torch.float, device=self.device)
        for _ in range(buffer_len):
            self.obs_buffer.append(obs_empty)

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
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    self.custom_torque_limits[i] = self.cfg.control.torque_limits[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                self.custom_torque_limits[i] = 100.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        #self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        #if self.cfg.domain_rand.randomize_gains:
        #    self.randomized_p_gains, self.randomized_d_gains, \
        #        self.randomized_motor_strength = self.compute_randomized_gains(self.num_envs)
            
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
        base_ang_vel = torch.tensor(low_state.imu_state.gyroscope, dtype=torch.float, device=self.device)
        projected_gravity = torch.tensor(low_state.imu_state.accelerometer, dtype=torch.float, device=self.device)

        dof_pos_list = [m.q for m in low_state.motor_state]
        dof_pos_list.pop(h1.H1JointIndex.kNotUsedJoint)
        self.dof_pos = torch.tensor(dof_pos_list, dtype=torch.float, device=self.device)

        dof_vel_list = [m.dq for m in low_state.motor_state]
        dof_vel_list.pop(h1.H1JointIndex.kNotUsedJoint)
        self.dof_vel = torch.tensor(dof_vel_list, dtype=torch.float, device=self.device)

        print("base_ang_vel shape:", base_ang_vel.shape)
        print("projected_gravity shape:", projected_gravity.shape)
        print("dof_pos shape:", self.dof_pos.shape)
        print("default_dof_pos shape:", self.default_dof_pos.shape)
        print("dof_vel shape:", self.dof_vel.shape)

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
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        torque_limits = self.custom_torque_limits

        return torch.clip(torques, -torque_limits, torque_limits)

    def run(self):
        while True:
            start = time.perf_counter()

            # read LowState
            low_state = self.h1_ctrl.low_state
            if low_state is None:
                time.sleep(0.001)
                continue

            # build & buffer observations
            obs = self.build_single_obs(low_state)
            self.obs_buffer.append(obs)

            # inference
            obs_stack = torch.stack(list(self.obs_buffer), dim=0)
            obs_batch = obs_stack.unsqueeze(0) 
            with torch.inference_mode():
                actions = self.policy.act_inference(obs_batch).squeeze(0)
            self.last_action = actions.clone()
            
            """# Upper‐body disturbances
            if self.use_disturb:
                # assume arm joints indices: e.g. 10–17
                arm_joints = list(range(10, 18))
                for j in arm_joints:
                    if self.disturb_mask[0, j] and self.disturb_isnoise:
                        # sample random torque scaled by curriculum radius
                        tau = (2*torch.rand(1, device=self.device)-1.0) \
                            * self.disturb_rad_curriculum
                        # add to your PD feedforward tau_ff
                        tau_ff[j] += tau.item()

            if self.standing_mask[0]:
                self.commands[0, :3] = 0.0"""
            
            clip_actions = self.cfg.normalization.clip_actions
            actions_clipped = torch.clip(actions.clone(), -clip_actions, clip_actions).to(self.device)
            torques = self.compute_torques(actions_clipped)
            
            # send LowCmd
            torques: list = torques.tolist()
            torques.insert(h1.H1JointIndex.kNotUsedJoint, 0.0)

            self.h1_ctrl.LowCmdWriteJointTorques(torques)

            # advance clock time
            self.t += self.dt

            # maintain real‐time
            elapsed = time.perf_counter() - start
            if self.dt - elapsed > 0.0:
                time.sleep(self.dt - elapsed)


if __name__ == "__main__":
    controller = Sim2Sim(
        device="cuda",
        domain_id=1,
        interface="lo"
    )
    controller.run()