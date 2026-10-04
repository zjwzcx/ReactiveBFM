import numpy as np
from scipy.spatial.transform import Rotation as R

POS_PICK = np.array([
    0,
    19,20,21,
    15,16,17,
    1,
    12,13,14,
    8,9,10
])

ROT_PICK = np.array([
    0,
    19,20,21,
    15,16,17,
    1,
    12,13,14,
    8,9,10
])

class XsensProcessor:
    # weishuai: A plug-and-use motion processor; You may consider using an online retargeter here;
    def __init__(self, scale_factor: float = 0.75, has_hand: bool = False):
        self.scale_factor = float(scale_factor)
        self.has_hand = has_hand

        # weishuai: hardcoded; xsens to unitree_g1
        self.left_shoulder = np.array([
            [1,0,0], [0,0,-1], [0,1,0]
        ])
        self.left_elbow = np.array([
            [0,0,1],[1,0,0], [0,1,0]
        ])
        self.left_wrist = np.array([
            [0,0,1],[1,0,0], [0,1,0]
        ])
        self.right_shoulder = np.array([
            [1,0,0], [0,0,1], [0,-1,0]
        ])
        self.right_elbow = np.array([
            [0,0,1],[-1,0,0], [0,-1,0]
        ])
        self.right_wrist = np.array([
            [0,0,1],[-1,0,0], [0,-1,0]
        ])
        
        self.link_rot_offset = np.stack([
            self.left_shoulder, self.left_elbow, self.left_wrist,
            self.right_shoulder, self.right_elbow, self.right_wrist
        ], axis=0)
        self.link_rot_offset = R.from_matrix(self.link_rot_offset)

        if self.has_hand:
            self._hand_limits_low = np.array([
                -1.04719755,-0.72431163,0,
                -1.57079632,-1.74532925,
                -1.57079632,-1.74532925,
                -1.04719755,-1.04719755,-1.74532925,
                0,0,
                0,0
            ])

            self._hand_limits_high = np.array([
                1.04719755,1.04719755,1.74532925,
                0,0,
                0,0,
                1.04719755,0.72431163,0,
                1.57079632,1.74532925,
                1.57079632,1.74532925
            ])

            self._hand_parent_idx = np.array([
                0,
                0,1,2,
                0,4,5,6,
                0,8,9,10,
                0,12,13,14,
                0,16,17,18,
                20,
                20,21,22,
                20,24,25,26,
                20,28,29,30,
                20,32,33,34,
                20,36,37,38
            ])
            self._hand_to_dex3_idx = np.array([
                1,2,3,
                9,10,
                5,6,
                21,22,23,
                29,30,
                25,26,
            ])
            self._dex3_qpos_scale = np.array([
                0.0, -2.625, -2.625,
                1.75, 1.75,
                1.75, 1.75,
                0.0, -2.625, -2.625,
                1.75, 1.75, 
                1.75, 1.75
            ])
            self._dex3_qpos_offset = np.array([
                1.04719755 / 2, 0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                1.04719755 / 2, 0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0
            ])
        else:
            self._dex3_default_qpos = np.array([
                1.04719755 / 2, 1.04719755,1.74532925,
                -1.57079632,-1.74532925,
                -1.57079632,-1.74532925,
                1.04719755 / 2, -1.04719755,-1.74532925,
                1.57079632,1.74532925,
                1.57079632,1.74532925
            ])
        
    def _hand_retargeting(self, ori_wxyz):
        # weishuai: hardcoded; xsens to dex3
        '''
        input:
            ori: np.array, shape (40, 4), xyzw
                index: 
                    0-19: left hand: left Carpus = 0
                                    first_finger_idx = [1,2,3]
                                    second_finger_idx = [4,5,6,7]
                                    third_finger_idx = [8,9,10,11]
                                    fourth_finger_idx = [12,13,14,15]
                                    fifth_finger_idx = [16,17,18,19]
                    20-39: right hand: right Carpus = 20
                                    first_finger_idx = [21, 22, 23]
                                    second_finger_idx = [24, 25, 26, 27]
                                    third_finger_idx = [28, 29, 30, 31]
                                    fourth_finger_idx = [32, 33, 34, 35]
                                    fifth_finger_idx = [36, 37, 38, 39]
        output:
            hand qpos: np.array, shape (14,) for G1 dex3 hand
        '''
        if self.has_hand:
            hand_ori = R.from_quat(ori_wxyz, scalar_first=True).as_matrix()        
            hand_ori_local = hand_ori[self._hand_parent_idx].transpose(0, 2, 1) @ hand_ori
            
            hand_qpos = R.from_matrix(hand_ori_local[self._hand_to_dex3_idx]).as_euler('xyz')[:, 0]
            hand_qpos = hand_qpos * self._dex3_qpos_scale + self._dex3_qpos_offset

            hand_qpos = np.clip(hand_qpos, self._hand_limits_low, self._hand_limits_high)
        else:
            hand_qpos = self._dex3_default_qpos.copy()

        return hand_qpos


    def process(self, pos: np.ndarray, ori_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected_pos = pos[POS_PICK,:]
        selected_ori_wxyz = ori_wxyz[ROT_PICK,:]

        selected_pos_scaled = selected_pos * self.scale_factor
        selected_ori_wxyz_normed = selected_ori_wxyz / np.linalg.norm(selected_ori_wxyz, axis=-1, keepdims=True).clip(min=1e-8)

        selected_ori_wxyz_normed[8:] = (R.from_quat(selected_ori_wxyz_normed[8:], scalar_first=True) * self.link_rot_offset).as_quat(scalar_first=True)

        special_links_parents = R.from_quat(selected_ori_wxyz_normed[[0,0]], scalar_first=True)
        special_links = R.from_quat(selected_ori_wxyz_normed[[1,4]], scalar_first=True)
        special_links = special_links_parents.inv() * special_links

        special_links = special_links.as_euler('YXZ')
        special_links[..., -1] *= 0 # remove yaw

        special_links = R.from_euler('YXZ', special_links)
        selected_ori_wxyz_normed[[1,4]] = (special_links_parents * special_links).as_quat(scalar_first=True)
        
        hand_qpos = self._hand_retargeting(ori_wxyz[23:])
        
        return selected_pos_scaled.astype(np.float32), selected_ori_wxyz_normed.astype(np.float32), hand_qpos.astype(np.float32)