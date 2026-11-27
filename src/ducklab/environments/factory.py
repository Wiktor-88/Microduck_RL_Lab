from pathlib import Path
import mujoco
import numpy as np
import torch
from ducklab.environments.rewards import simple_reward

# Self-contained 14-DOF, ~800g, ~25cm Microduck MJCF model
MICRODUCK_XML = """
<mujoco model="microduck">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" gravity="0 0 -9.81"/>

  <default>
    <joint armature="0.01" damping="0.25" limited="true"/>
    <geom friction="0.8 0.005 0.0001" rgba="0.95 0.75 0.1 1"/>
  </default>

  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="25 25 0.1" rgba="0.88 0.88 0.88 1"/>

    <body name="torso" pos="0 0 0.22">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="0.045 0.04 0.055" mass="0.45"/>
      <site name="imu" pos="0 0 0"/>

      <!-- Left Leg (5 DOF: indices 0..4) -->
      <body name="left_hip" pos="0 0.035 -0.055">
        <joint name="left_hip_yaw" axis="0 0 1" range="-0.5 0.5"/>
        <joint name="left_hip_roll" axis="1 0 0" range="-0.4 0.4"/>
        <joint name="left_hip_pitch" axis="0 1 0" range="-1.0 1.0"/>
        <geom name="left_thigh" type="capsule" fromto="0 0 0 0 0 -0.06" size="0.015" mass="0.06"/>
        <body name="left_shin" pos="0 0 -0.06">
          <joint name="left_knee" axis="0 1 0" range="-1.2 0.2"/>
          <geom name="left_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.06" size="0.013" mass="0.05"/>
          <body name="left_foot" pos="0 0 -0.06">
            <joint name="left_ankle" axis="0 1 0" range="-0.6 0.6"/>
            <geom name="left_foot_geom" type="box" pos="0.015 0 -0.01" size="0.045 0.025 0.01" mass="0.04"/>
          </body>
        </body>
      </body>

      <!-- Head & Neck (4 DOF: indices 5..8) -->
      <body name="neck" pos="0.02 0 0.055">
        <joint name="neck_pitch" axis="0 1 0" range="-0.5 0.5"/>
        <geom name="neck_geom" type="capsule" fromto="0 0 0 0 0 0.03" size="0.01" mass="0.03"/>
        <body name="head" pos="0 0 0.03">
          <joint name="head_pitch" axis="0 1 0" range="-0.5 0.5"/>
          <joint name="head_yaw" axis="0 0 1" range="-0.8 0.8"/>
          <joint name="head_roll" axis="1 0 0" range="-0.4 0.4"/>
          <geom name="head_geom" type="sphere" pos="0.015 0 0.02" size="0.035" mass="0.08"/>
        </body>
      </body>

      <!-- Right Leg (5 DOF: indices 9..13) -->
      <body name="right_hip" pos="0 -0.035 -0.055">
        <joint name="right_hip_yaw" axis="0 0 1" range="-0.5 0.5"/>
        <joint name="right_hip_roll" axis="1 0 0" range="-0.4 0.4"/>
        <joint name="right_hip_pitch" axis="0 1 0" range="-1.0 1.0"/>
        <geom name="right_thigh" type="capsule" fromto="0 0 0 0 0 -0.06" size="0.015" mass="0.06"/>
        <body name="right_shin" pos="0 0 -0.06">
          <joint name="right_knee" axis="0 1 0" range="-1.2 0.2"/>
          <geom name="right_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.06" size="0.013" mass="0.05"/>
          <body name="right_foot" pos="0 0 -0.06">
            <joint name="right_ankle" axis="0 1 0" range="-0.6 0.6"/>
            <geom name="right_foot_geom" type="box" pos="0.015 0 -0.01" size="0.045 0.025 0.01" mass="0.04"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="a_l_hip_yaw" joint="left_hip_yaw" kp="4.0" ctrlrange="-0.5 0.5"/>
    <position name="a_l_hip_roll" joint="left_hip_roll" kp="4.0" ctrlrange="-0.4 0.4"/>
    <position name="a_l_hip_pitch" joint="left_hip_pitch" kp="5.0" ctrlrange="-1.0 1.0"/>
    <position name="a_l_knee" joint="left_knee" kp="5.0" ctrlrange="-1.2 0.2"/>
    <position name="a_l_ankle" joint="left_ankle" kp="4.0" ctrlrange="-0.6 0.6"/>
    <position name="a_neck_pitch" joint="neck_pitch" kp="2.0" ctrlrange="-0.5 0.5"/>
    <position name="a_head_pitch" joint="head_pitch" kp="2.0" ctrlrange="-0.5 0.5"/>
    <position name="a_head_yaw" joint="head_yaw" kp="2.0" ctrlrange="-0.8 0.8"/>
    <position name="a_head_roll" joint="head_roll" kp="2.0" ctrlrange="-0.4 0.4"/>
    <position name="a_r_hip_yaw" joint="right_hip_yaw" kp="4.0" ctrlrange="-0.5 0.5"/>
    <position name="a_r_hip_roll" joint="right_hip_roll" kp="4.0" ctrlrange="-0.4 0.4"/>
    <position name="a_r_hip_pitch" joint="right_hip_pitch" kp="5.0" ctrlrange="-1.0 1.0"/>
    <position name="a_r_knee" joint="right_knee" kp="5.0" ctrlrange="-1.2 0.2"/>
    <position name="a_r_ankle" joint="right_ankle" kp="4.0" ctrlrange="-0.6 0.6"/>
  </actuator>

  <sensor>
    <gyro name="imu_gyro" site="imu"/>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
  </sensor>
</mujoco>
"""


class MockEnv:
    """
    Real MuJoCo rigid-body physics environment for Microduck (50 Hz control,
    14-DOF position actuators, 61D proprioceptive observation contract).
    """

    def __init__(
        self,
        num_envs: int = 100, # how many robots at once
        obs_dim: int = 61, #sensors
        action_dim: int = 14, #joints
        max_steps: int = 200, 
        substeps: int = 5,  # 5 * 0.004s = 0.02s (50 Hz)
        xml_path: str | None = None, #for robot configuration
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.substeps = substeps

        if xml_path and Path(xml_path).exists():
            self.model = mujoco.MjModel.from_xml_path(xml_path) # type: ignore
        else:
            self.model = mujoco.MjModel.from_xml_string(MICRODUCK_XML) # type: ignore

        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)] # whats currently happening
        self.reset() #initialise

    def reset(self) -> torch.Tensor:
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long) #how long did each robot survivr
        self.prev_actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32) #previous action
        self.vel_cmd = np.zeros((self.num_envs, 3), dtype=np.float32) #target movement
        self.vel_cmd[:, 0] = 0.3  # [0] - x axis, we want it to move 0.3 m/s forward

        for d in self.datas:
            mujoco.mj_resetData(self.model, d) #put robots into initial state
            # Small random initial joint perturbation
            d.qpos[7 : 7 + self.action_dim] = np.random.uniform(-0.03, 0.03, size=self.action_dim)
            mujoco.mj_forward(self.model, d) # recalculate positions

        return self._get_obs()

    def _get_obs(self) -> torch.Tensor:
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        for i, d in enumerate(self.datas):
            # [0:3] IMU gyroscope (angular velocity)
            obs[i, 0:3] = d.sensordata[0:3]

            # [3:6] Projected gravity from IMU quaternion (qw, qx, qy, qz)
            qw, qx, qy, qz = d.sensordata[3:7]
            obs[i, 3] = -2.0 * (qx * qz - qw * qy)
            obs[i, 4] = -2.0 * (qy * qz + qw * qx)
            obs[i, 5] = -(1.0 - 2.0 * (qx * qx + qy * qy))

            # [6:20] Joint positions (14 DOF)
            obs[i, 6:20] = d.qpos[7 : 7 + self.action_dim]

            # [20:34] Joint velocities (14 DOF)
            obs[i, 20:34] = d.qvel[6 : 6 + self.action_dim]

            # [34:48] Previous action (14 DOF)
            obs[i, 34:48] = self.prev_actions[i]

            # [48:51] Target velocity command
            obs[i, 48:51] = self.vel_cmd[i]

        return torch.from_numpy(obs)

    def step(self, actions: torch.Tensor):
        act_np = torch.clamp(actions, -1.0, 1.0).cpu().numpy()
        self.step_count += 1

        vel_x = np.zeros(self.num_envs, dtype=np.float32)
        pos_x = np.zeros(self.num_envs, dtype=np.float32)
        fallen = np.zeros(self.num_envs, dtype=bool)

        for i, d in enumerate(self.datas):
            d.ctrl[: self.action_dim] = act_np[i]
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, d) # type: ignore

            pos_x[i] = d.qpos[0]
            vel_x[i] = d.qvel[0]
            torso_z = d.qpos[2]

            qw, qx, qy, _ = d.sensordata[3:7]
            proj_grav_z = -(1.0 - 2.0 * (qx * qx + qy * qy))
            # Robot fell if torso drops below 0.12 m or tilts > 60 deg
            fallen[i] = (torso_z < 0.12) or (proj_grav_z > -0.5)

        self.prev_actions = act_np.copy()

        vel_x_t = torch.from_numpy(vel_x)
        fallen_t = torch.from_numpy(fallen)
        timeout_t = self.step_count >= self.max_steps
        dones = fallen_t | timeout_t

        rewards = simple_reward(vel_x_t, torch.from_numpy(act_np), fallen_t)

        info = {
            "pos_x": torch.from_numpy(pos_x),
            "vel_x": vel_x_t,
            "fallen": fallen_t,
            "steps": self.step_count.clone(),
        }
        return self._get_obs(), rewards, dones, info