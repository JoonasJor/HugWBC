import os
import sys


sys.path.append(os.getcwd())
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils import get_args, task_registry, update_class_from_dict
#from legged_gym.utils.custom_eval import CustomEval
from isaacgym import gymapi
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
from isaacgym import gymapi
import h1_low_level_control as h1
import legged_gym.legged_utils

def play(args):
    device = "cuda"
    h1.ChannelFactoryInitialize(1, "lo")

    h1_ctrl = h1.LowLevelControl()
    h1_ctrl.Init()
    #h1_ctrl.LowCmdWriteZero()
    #custom.Start()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    resume_path = train_cfg.runner.resume_path
    print(resume_path)

    # prepare     # planeenvironment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    for i in range(env.num_bodies):
        env.gym.set_rigid_body_color(env.envs[0], env.actor_handles[0], i, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.3))
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    _, _ = env.reset()
    obs, critic_obs, _, _, _ = env.step(torch.zeros(
            env.num_envs, env.num_actions, dtype=torch.float, device=env.device))
    
    print(f"obs shape initial: {obs.shape}")

    
    commands = torch.zeros(11, dtype=torch.float, device=device)
    obs = torch.zeros((5,76), dtype=torch.float, device=device)
    step_count = 0

    obs_scales = LeggedRobotCfg.normalization.obs_scales()
    
    while True:
        with torch.inference_mode():
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

            # Mujoco proprioception
            low_state = h1_ctrl.low_state
            print(low_state)
            if low_state != None:
                angular_velocity = low_state.imu_state.gyroscope
                gravity_projection = low_state.imu_state.accelerometer
                joint_position = [motor.q for motor in low_state.motor_state]
                joint_position.pop(h1.H1JointIndex.kNotUsedJoint)
                joint_velocity = [motor.dq for motor in low_state.motor_state]
                joint_velocity.pop(h1.H1JointIndex.kNotUsedJoint)
                #print(len(angular_velocity) + len(gravity_projection) + len(joint_position) + len(joint_velocity))
            
            
            # Lower body movement commands
            commands[0] = 0.8    # Linear velocity X ( m/s )     ( min: -0.6, max: 2.0 )
            commands[1] = 0.0    # Linear velocity Y ( m/s )     ( min: -0.6, max: 0.6 )
            commands[2] = 0.0    # Angular velocity  ( rad/s )   ( min: -1.0, max: 1.0 )

            commands[3] = 2.0    # Gait frequency    ( Hz )      ( min: 1.5, max: 3.5 )
            commands[4] = 0.5    # Phase offset                  ( 0.5 = walking, 1.0 = jumping )

            commands[5] = 0.5    # Body height       ( m )       ( min: -0.3, max: 0.5 )
            commands[6] = 0.0    # ? 
            commands[7] = 0.0    # ?

            commands[8] = 0.0    # Body pitch        ( rad )     ( min: 0.0, max: 0.4 )
            commands[9] = 0.0    # Waist rotation    ( rad )     ( min: ?, max: ? )
            commands[10] = 0.0   # ?

            single_obs = torch.zeros((76,), dtype=torch.float, device=device)
            # Fill proprioception (44)
            single_obs[0:3] = torch.tensor(angular_velocity, device="cuda")
            single_obs[3:6] = torch.tensor(gravity_projection, device="cuda") 
            single_obs[6:25] = torch.tensor(joint_position, device="cuda")
            single_obs[25:44] = torch.tensor(joint_velocity, device="cuda")

            # Fill last actions (19) — store the previous one if available
            if 'last_action' in locals():
                single_obs[44:63] = last_action.clone()
            else:
                single_obs[44:63] = torch.zeros((19,), device=device)

            # Fill last commands (11)
            single_obs[63:74] = commands.clone()

            # Commands (11) with appropriate scaling
            cmd_scaled = torch.zeros_like(commands)
            cmd_scaled[0] = commands[0] * obs_scales.lin_vel       # lin vel x
            cmd_scaled[1] = commands[1] * obs_scales.lin_vel       # lin vel y
            cmd_scaled[2] = commands[2] * obs_scales.ang_vel       # ang vel
            cmd_scaled[3] = commands[3] * obs_scales.gait_freq_cmd
            cmd_scaled[4] = commands[4] * obs_scales.gait_phase_cmd
            cmd_scaled[5] = commands[5] * obs_scales.body_height_cmd
            cmd_scaled[6] = commands[6] * obs_scales.footswing_height_cmd
            cmd_scaled[7] = commands[7] * obs_scales.height_measurements
            cmd_scaled[8] = commands[8] * obs_scales.body_pitch_cmd
            cmd_scaled[9] = commands[9] * obs_scales.waist_roll_cmd
            cmd_scaled[10] = commands[10]  # no scale defined

            # Apply scaled commands
            single_obs[63:74] = cmd_scaled

            # Fill clock (2) — example clock input; replace with yours
            single_obs[74:76] = torch.tensor([step_count % 1000 / 1000.0, step_count % 2], device=device)

            # 2. Insert into rolling buffer
            obs = torch.roll(obs, shifts=-1, dims=0)  # shift buffer up
            obs[-1] = single_obs  # insert newest obs

            # 3. Store current action before computing new one
            actions = policy.act_inference(obs)
            last_action = actions[0].detach()

            # Increment step count (needed for clock logic)
            step_count += 1

            # Mujoco joint control
            joint_angles: list = actions[0].flatten().tolist()
            joint_angles.insert(h1.H1JointIndex.kNotUsedJoint, 0.0)

            h1_ctrl.LowCmdWriteJointTorques(joint_angles)

if __name__ == '__main__':
    args = get_args()
    play(args)