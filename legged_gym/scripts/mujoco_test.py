import os
import sys
sys.path.append(os.getcwd())

from legged_gym import LEGGED_GYM_ROOT_DIR
import argparse
import numpy as np
import torch
from mujoco import MjModel, MjData, mj_step, mj_reset
from mujoco.viewer import MjViewer

# Helpers from HugWBC / legged_gym
#from legged_gym.utils.obs_utils import (
#    compute_observation,
#    compute_privileged_observation
#)
#from legged_gym.utils.action_utils import map_actions_to_pd

# Optional: your disturbance code
def apply_random_push(data, model, force_rad):
    # apply a random planar push at the torso
    body_id = model.body_name2id['torso']
    dir_angle = np.random.uniform(0, 2*np.pi)
    force = force_rad * np.array([np.cos(dir_angle), np.sin(dir_angle), 0.0])
    data.xfrc_applied[body_id, :3] += force

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pt', required=True)
    parser.add_argument('--scene', default='resources/robots/h1/h1_scene.xml')
    args = parser.parse_args()

    # Load MuJoCo model and data
    model = MjModel.from_xml_path(args.scene)
    data  = MjData(model)
    viewer = MjViewer(model, data)

    # Load policy
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    policy = torch.jit.load(args.model_pt).to(device)
    policy.eval()

    # Reset to default pose
    mj_reset(model, data)
    for _ in range(100): mj_step(model, data)

    # Preallocate command array (1 env)
    commands = np.zeros((1,10), dtype=np.float32)
    commands[0] = [2.0, 0, 0, 2.0, 0.5, 0.5, 0.2, 0.0, 0.0, 0.0]

    use_disturb   = True
    disturb_force = 1.0
    cam_rot       = 0.0
    cam_rate      = 0.2
    viewer_dt     = 0.02
    next_view_t   = viewer_dt
    t             = 0.0

    while True:
        # 1) Physics step
        mj_step(model, data)
        t += model.opt.timestep

        # 2) Build obs / priv_obs
        qpos = torch.from_numpy(data.qpos[None]).float().to(device)
        qvel = torch.from_numpy(data.qvel[None]).float().to(device)

        obs      = compute_observation(qpos, qvel, torch.from_numpy(commands))
        priv_obs = compute_privileged_observation(qpos, qvel)

        # 3) Inference
        with torch.inference_mode():
            actions = policy.act_inference(obs, privileged_obs=priv_obs)

        # 4) Map to PD commands & apply
        q_des, dq_des, tau_ff = map_actions_to_pd(actions.cpu().numpy())
        # Example PD control: torque = Kp*(q_des - qpos) + Kd*(dq_des - qvel) + tau_ff
        data.ctrl[:] = tau_ff  # or replace with full PD

        # 5) Disturbances
        if use_disturb and np.random.rand() < 0.005:
            apply_random_push(data, model, disturb_force)

        # 6) Camera orbit (slow-rate)
        if t >= next_view_t:
            cam_rot = (cam_rot + cam_rate * model.opt.timestep) % (2*np.pi)
            look_at = data.xpos[model.body_name2id['torso'], :3]
            eye     = look_at + np.array([
                np.cos(cam_rot)*1.0,
                np.sin(cam_rot)*1.0,
                0.5*0.8
            ])
            viewer.cam.lookat[:] = look_at
            viewer.cam.distance  = np.linalg.norm(eye - look_at)
            next_view_t += viewer_dt

        # 7) Render
        viewer.render()

if __name__ == '__main__':
    main()
