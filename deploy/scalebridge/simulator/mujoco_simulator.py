import os
import time
from pathlib import Path
import torch
import imageio
import threading
import mujoco
import mujoco.viewer as mjv
import numpy as np
from scalebridge.simulator.base_simulator import BaseSimulator
from scalebridge.utils.logging import logger
from scalebridge.utils.merge_robot_object_xml import merge_robot_object_xml
from scalebridge.utils.overlay import load_font, overlay_header

def draw_marker(pos,v):
    geom = v.user_scn.geoms[v.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.03,0.03,0.03],
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=[1,0,0,1]
    )
    v.user_scn.ngeom += 1

class MujocoSimulator(BaseSimulator):
    def __init__(self, config, metadata_dict):
        
        self.record_video = config.get('record_video', False)
        self.video_path = config.get('video_path', None)
        self.render_width = int(config.get('render_width', 640))
        self.render_height = int(config.get('render_height', 480))
        self.video_text = str(config.get('video_text', ''))
        self.video_total_frames = int(config.get('video_total_frames', 0))
        self.video_font_size = int(config.get('video_font_size', 20))
        self.video_frame_idx = 0
        self.headless = config.get('headless', False)
        self.marker = config.get('marker', False)
        self.use_joystick = config.get('joystick', False)
        self.camera_follow = config.get('camera_follow', False) or self.record_video

        super().__init__(config, metadata_dict)

        self._setup_joystick()

    def _setup_backbone(self):
        super()._setup_backbone()

        xml_path = self.cfg.asset.xml_path

        # pase object from metadata if exists
        self.has_object = "object_names" in self.metadata_dict
        if self.has_object:
            xml_path = merge_robot_object_xml(xml_path, self.metadata_dict)

        self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        self.mujoco_model.opt.timestep=self.low_dt
        
        self.viewer = None
        if not self.headless:
            self.viewer = mjv.launch_passive(
                model=self.mujoco_model,
                data=self.mujoco_data,
                show_left_ui=False,
                show_right_ui=False,
            )
        self.marker_pos = None
        if self.record_video:
            if self.video_path:
                video_name = str(Path(self.video_path).expanduser().resolve())
            else:
                video_name = os.path.join(os.getcwd(), 'recording.mp4')
            Path(video_name).parent.mkdir(parents=True, exist_ok=True)
            self.video_path = video_name
            self.video_fps = 1.0 / self.high_dt
            self.video_writer = imageio.get_writer(video_name, fps=self.video_fps)
            self.renderer = mujoco.Renderer(
                self.mujoco_model,
                height=self.render_height,
                width=self.render_width,
            )
            self.render_scene = mujoco.MjvScene(self.mujoco_model, maxgeom=1000)
            self.render_camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.mujoco_model, self.render_camera)
            self.video_font = load_font(self.video_font_size)
            logger.info(f"[Simulator] Recording video to {video_name}")


    def _setup_asset(self):
        super()._setup_asset()

        self.default_qpos = self.mujoco_data.qpos.copy()
        self.default_qvel = self.mujoco_data.qvel.copy()

        self.default_dof_pos = self.mujoco_data.qpos[7:7+self.num_joints].copy()

    def _setup_joystick(self):
        if self.use_joystick:
            import pygame
            logger.info(f'[Simulator] Using Joystick as command sender; Make sure you have connected the joystick to the PC!')
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                logger.info(f"[Simulator] Joystick detected: {self.joystick.get_name()}")
            else:
                pygame.quit()
                logger.error(f"[Simulator] Joystick undetected!")
            self.lin_vel_x_tmp = 0
            self.lin_vel_y_tmp = 0
            self.ang_vel_z_tmp = 0
            self.joystick_thread = threading.Thread(target=self._handle_joystick, args=(pygame,),daemon=True)
            self.joystick_thread.start()
    
    def _handle_joystick(self, pygame):
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        break
                self.lin_vel_x_tmp = self.joystick.get_axis(1) * -1
                self.lin_vel_y_tmp = self.joystick.get_axis(0) * -1
                self.ang_vel_z_tmp = self.joystick.get_axis(3) * -1
                time.sleep(0.001)
        except:
            pygame.quit()

    def _render(self):
        if self.headless and not self.record_video:
            return
        if not self.headless and not self.viewer:
            logger.info(f"[Simulator] Viewer has been closed; Automatically exit!")
            exit()

        if self.viewer and self.marker_pos is not None and self.marker:
            self.viewer.user_scn.ngeom = 0
            for i in range(self.marker_pos.shape[0]):
                draw_marker(
                    self.marker_pos[i], self.viewer
                )
            if self.record_video:
                self.render_scene.ngeom = 0
                for i in range(self.marker_pos.shape[0]):
                    geom = self.render_scene.geoms[self.render_scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.03, 0.03, 0.03],
                        pos=self.marker_pos[i],
                        mat=np.eye(3).flatten(),
                        rgba=[1,0,0,1]
                    )
                    self.render_scene.ngeom += 1
        camera = self.viewer.cam if self.viewer else getattr(self, "render_camera", None)
        if camera is not None and self.camera_follow:
            camera.lookat = self.mujoco_data.qpos[:3].copy()
            camera.elevation = 0
            camera.azimuth = 180
            camera.distance = 3.0

        if self.viewer:
            self.viewer.sync()

        if self.record_video:
            self.renderer.update_scene(self.mujoco_data, camera=camera)
        
            # Then manually add markers to the renderer's scene
            if self.marker_pos is not None and self.marker:
                # The renderer's scene might need to be updated after update_scene
                # So we add markers after the update
                for i in range(self.marker_pos.shape[0]):
                    geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.03, 0.03, 0.03],
                        pos=self.marker_pos[i],
                        mat=np.eye(3).flatten(),
                        rgba=[1,0,0,1]
                    )
                    self.renderer.scene.ngeom += 1

            img = self.renderer.render()
            img = overlay_header(
                img,
                frame_idx=self.video_frame_idx,
                total_frames=max(self.video_total_frames, self.video_frame_idx + 1),
                fps=self.video_fps,
                text_label=self.video_text,
                font=self.video_font,
            )
            self.video_writer.append_data(img)
            self.video_frame_idx += 1

    def close(self):
        if getattr(self, "video_writer", None) is not None:
            self.video_writer.close()
            self.video_writer = None
        if getattr(self, "renderer", None) is not None:
            self.renderer.close()
            self.renderer = None
        if getattr(self, "viewer", None) is not None:
            self.viewer.close()
            self.viewer = None

    def update_marker_pos(self, marker_pos):
        self.marker_pos = marker_pos.cpu().numpy()

    def refresh_sim(self):

        state_dict = { # weishuai: FIXME
            "root_pos": self.mujoco_data.qpos[:3],
            "root_quat_wxyz": self.mujoco_data.qpos[3:7],
            "base_ang_vel": self.mujoco_data.qvel[3:6],
            "dof_pos": self.mujoco_data.qpos[7:7+self.num_joints][self.sim_to_env_joint_idx],
            "dof_vel": self.mujoco_data.qvel[6:6+self.num_joints][self.sim_to_env_joint_idx],
        }

        if self.has_object:
            object_root_pose = self.mujoco_data.qpos[7+self.num_joints:].reshape(-1, 7)
            state_dict.update({
                "object_root_pos": object_root_pose[:, :3],
                "object_root_quat_wxyz": object_root_pose[:, 3:7],
            })

        if self.use_joystick:
            self.commands = np.array([self.lin_vel_x_tmp, self.lin_vel_y_tmp, self.ang_vel_z_tmp], dtype=np.float32)
            self.commands = np.where(np.abs(self.commands) < 0.05, 0, self.commands)
            state_dict.update({'commands': self.commands})
        
        return {k:torch.from_numpy(v).float() for k,v in state_dict.items()}

    def calibrate(self, init_state_dict = {}):
        # Internal interface: quaternions are wxyz (MuJoCo qpos order).
        # Planner-facing xyzw inputs are converted in BaseEnv.reset.
        init_qpos = self.default_qpos.copy()
        init_qvel = self.default_qvel.copy()

        if "root_pos" in init_state_dict:
            init_qpos[:3] = init_state_dict["root_pos"]
        if "root_quat" in init_state_dict:
            init_qpos[3:7] = init_state_dict["root_quat"]
        if "dof_pos" in init_state_dict:
            init_qpos[7:7+self.num_joints][self.env_action_to_sim_idx] = init_state_dict["dof_pos"]
        if "root_lin_vel" in init_state_dict:
            init_qvel[:3] = init_state_dict["root_lin_vel"]
        if "root_ang_vel" in init_state_dict:
            init_qvel[3:6] = init_state_dict["root_ang_vel"]
        if "dof_vel" in init_state_dict:
            init_qvel[6:6+self.num_joints][self.env_action_to_sim_idx] = init_state_dict["dof_vel"]
        
        if self.has_object:
            
            init_obj_pose = init_qpos[7+self.num_joints:].copy().reshape(-1, 7)
            init_obj_vel = init_qvel[6+self.num_joints:].copy().reshape(-1, 6)
            
            if "object_root_pos" in init_state_dict:
                init_obj_pose[:, :3] = init_state_dict["object_root_pos"]
            if "object_root_quat" in init_state_dict:
                init_obj_pose[:, 3:7] = init_state_dict["object_root_quat"]
            if "object_root_lin_vel" in init_state_dict:
                init_obj_vel[:, :3] = init_state_dict["object_root_lin_vel"]
            if "object_root_ang_vel" in init_state_dict:
                init_obj_vel[:, 3:] = init_state_dict["object_root_ang_vel"]

            init_qpos[7+self.num_joints:] = init_obj_pose.flatten()
            init_qvel[6+self.num_joints:] = init_obj_vel.flatten()

        self.mujoco_data.qpos[:] = init_qpos
        self.mujoco_data.qvel[:] = init_qvel
        self.mujoco_data.ctrl[:] = 0

        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)

        return init_qpos[:3], init_qpos[3:7]

    def apply_action(self, tgt_dof_pos):
        tgt_dof_pos = tgt_dof_pos.squeeze()

        target_dof_pos_in_sim = self.default_dof_pos.copy()
        target_dof_pos_in_sim[self.env_action_to_sim_idx] = tgt_dof_pos

        for _ in range(self.decimation):
            torque = (target_dof_pos_in_sim - self.mujoco_data.qpos[7:7+self.num_joints]) * self.stiffness - self.mujoco_data.qvel[6:6+self.num_joints] * self.damping # no clip here
            # torque = np.clip(torque, -self.torque_limit, self.torque_limit)
            self.mujoco_data.ctrl[:] = torque # weishuai: no clip applied here
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

        self._render()

    def _get_joint_names(self):
        joint_names = []
        for i in range(self.mujoco_model.nu):
            dof_name = mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            joint_names.append(dof_name)
        return joint_names
