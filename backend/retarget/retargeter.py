"""Motion retargeting core: retargets SourceMotion (FBX/BVH) to PMXModel and builds VMDMotion."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation

from ..skeleton import SourceMotion
from ..parser.pmx_parser import PMXModel
from ..exporter.vmd_exporter import VMDMotion, BoneTrack
from .bone_mapping import SourceMap, TargetMap, map_source, map_target, SIDES, FINGERS, _key
from .math_utils import (
    MIRROR_Z,
    normalize,
    orthonormalize,
    rotation_between,
    mat_to_quat,
    quat_to_mat,
    quat_make_continuous,
    resample_globals,
    swing_twist,
)
from .ik_solver import forward_kinematics, solve_ik_offsets


@dataclass
class RetargetConfig:
    enable_foot_ik: bool = True
    enable_twist_bones: bool = True
    enable_fingers: bool = True
    scale_multiplier: float = 1.0
    auto_detect_a_pose: bool = True
    target_fps: float = 30.0
    overrides: dict[str, str] = field(default_factory=dict)


class Retargeter:
    def __init__(self, source: SourceMotion, target: PMXModel, config: RetargetConfig | None = None):
        self.source = source
        self.target = target
        self.cfg = config or RetargetConfig()
        self.src_map: SourceMap = map_source(self.source, self.cfg.overrides)
        self.tgt_map: TargetMap = map_target(self.target)

    def calculate_scale(self) -> float:
        """Calculate leg length ratio (target_leg / source_leg) to absorb stature differences."""
        # Source leg length (average L and R)
        src_lens = []
        for sd in SIDES:
            ul = self.src_map.get(_key("upperLeg", sd))
            ll = self.src_map.get(_key("lowerLeg", sd))
            ft = self.src_map.get(_key("foot", sd))
            if ul >= 0 and ll >= 0 and ft >= 0:
                l1 = np.linalg.norm(self.source.rest_pos[ll] - self.source.rest_pos[ul])
                l2 = np.linalg.norm(self.source.rest_pos[ft] - self.source.rest_pos[ll])
                src_lens.append(l1 + l2)

        # Target leg length (average L and R)
        tgt_lens = []
        for sd in SIDES:
            ul = self.tgt_map.get(_key("upperLeg", sd))
            ll = self.tgt_map.get(_key("lowerLeg", sd))
            ft = self.tgt_map.get(_key("foot", sd))
            if ul >= 0 and ll >= 0 and ft >= 0:
                p_ul = self.target.bones[ul].position
                p_ll = self.target.bones[ll].position
                p_ft = self.target.bones[ft].position
                l1 = np.linalg.norm(p_ll - p_ul)
                l2 = np.linalg.norm(p_ft - p_ll)
                tgt_lens.append(l1 + l2)

        if src_lens and tgt_lens:
            s_src = float(np.mean(src_lens))
            s_tgt = float(np.mean(tgt_lens))
            if s_src > 1e-4:
                return (s_tgt / s_src) * self.cfg.scale_multiplier

        return 1.0 * self.cfg.scale_multiplier

    def retarget(self) -> VMDMotion:
        # 1. Resample source motion to 30fps
        rot, pos = resample_globals(
            self.source.rot, self.source.pos, self.source.fps, self.cfg.target_fps
        )
        n_frames = rot.shape[0]
        scale = self.calculate_scale()

        # 2. Coordinate conversion: Right-Handed (Source) to Left-Handed MMD (Target)
        # R_lh = Mz @ R_rh @ Mz
        # pos_lh = pos_rh * [1, 1, -1]
        rot_lh = np.einsum("ia,fjab,bk->fjik", MIRROR_Z, rot, MIRROR_Z)
        pos_lh = pos * np.array([1.0, 1.0, -1.0])
        rest_rot_lh = np.einsum("ia,jab,bk->jik", MIRROR_Z, self.source.rest_rot, MIRROR_Z)
        rest_pos_lh = self.source.rest_pos * np.array([1.0, 1.0, -1.0])

        tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # bone_name -> (pos_array, rot_quat_array)

        # 3. Rest pose offset analysis (A-pose to T-pose compensation)
        a_pose_offsets: dict[str, np.ndarray] = {}  # semantic -> 3x3 rotation offset
        if self.cfg.auto_detect_a_pose:
            self._compute_a_pose_offsets(rest_pos_lh, a_pose_offsets)

        # 4. Hips / Center / Root handling
        hips_idx = self.src_map.get("hips")
        center_pmx = self.tgt_map.get("center")
        groove_pmx = self.tgt_map.get("groove")

        if hips_idx >= 0 and center_pmx >= 0:
            # Root displacement scaled to PMX stature
            d_pos = (pos_lh[:, hips_idx] - rest_pos_lh[hips_idx]) * scale
            # Hips delta orientation
            hips_delta_rot = rot_lh[:, hips_idx] @ np.linalg.inv(rest_rot_lh[hips_idx])

            center_name = self.target.bones[center_pmx].name
            # In MMD, "センター" usually carries position + rotation, or position only if "下半身" takes rotation
            # We assign position to センター, and rotation to センター or 下半身
            hips_pmx = self.tgt_map.get("hips")
            if hips_pmx >= 0:
                # センター carries translation and forward orientation, 下半身 carries lower body rotation
                tracks[center_name] = (d_pos, mat_to_quat(np.broadcast_to(np.eye(3), (n_frames, 3, 3))))
                hips_name = self.target.bones[hips_pmx].name
                tracks[hips_name] = (np.zeros((n_frames, 3)), mat_to_quat(hips_delta_rot))
            else:
                tracks[center_name] = (d_pos, mat_to_quat(hips_delta_rot))

        # 5. Spine chain distribution
        self._retarget_spine(rot_lh, rest_rot_lh, n_frames, tracks)

        # 6. Head & Neck
        for sem in ("neck", "head"):
            s_idx = self.src_map.get(sem)
            t_idx = self.tgt_map.get(sem)
            if s_idx >= 0 and t_idx >= 0:
                p_s = self.source.parents[s_idx]
                if p_s >= 0:
                    r_cur = np.linalg.inv(rot_lh[:, p_s]) @ rot_lh[:, s_idx]
                    r_rest = np.linalg.inv(rest_rot_lh[p_s]) @ rest_rot_lh[s_idx]
                    local_r = r_cur @ np.linalg.inv(r_rest)
                else:
                    local_r = rot_lh[:, s_idx] @ np.linalg.inv(rest_rot_lh[s_idx])
                bname = self.target.bones[t_idx].name
                tracks[bname] = (np.zeros((n_frames, 3)), mat_to_quat(local_r))

        # 7. Arms and Limbs
        for sd in SIDES:
            self._retarget_arm(sd, rot_lh, rest_rot_lh, a_pose_offsets, n_frames, tracks)
            self._retarget_leg(sd, rot_lh, rest_rot_lh, pos_lh, rest_pos_lh, scale, n_frames, tracks)

        # 8. Fingers (Phase 1)
        if self.cfg.enable_fingers:
            self._retarget_fingers(rot_lh, rest_rot_lh, n_frames, tracks)

        # 9. Build VMDMotion object
        vmd = VMDMotion(model_name=self.target.name or "MMDModel")
        frames_arr = np.arange(n_frames, dtype=int)
        for bname, (pos_arr, rot_quat) in tracks.items():
            rot_quat = quat_make_continuous(rot_quat)
            vmd.bones.append(
                BoneTrack(
                    name=bname,
                    frames=frames_arr,
                    positions=pos_arr,
                    rotations=rot_quat,
                )
            )

        # Add default IK enable states if foot IK was generated
        if self.cfg.enable_foot_ik:
            ik_names = []
            for sd in SIDES:
                for k in ("legIK", "toeIK"):
                    b_idx = self.tgt_map.get(_key(k, sd))
                    if b_idx >= 0:
                        ik_names.append((self.target.bones[b_idx].name, True))
            if ik_names:
                vmd.ik_states.append((0, ik_names))

        return vmd

    def _compute_a_pose_offsets(self, rest_pos_lh: np.ndarray, out_offsets: dict[str, np.ndarray]):
        """Detect A-pose in upper arms and compute compensation to T-pose (horizontal)."""
        for sd in SIDES:
            ua = self.src_map.get(_key("upperArm", sd))
            la = self.src_map.get(_key("lowerArm", sd))
            if ua >= 0 and la >= 0:
                v_src = normalize(rest_pos_lh[la] - rest_pos_lh[ua])
                # Target T-pose arm vector in MMD: Left arm goes +X [1, 0, 0], Right arm goes -X [-1, 0, 0]
                v_tgt = np.array([1.0 if sd == "L" else -1.0, 0.0, 0.0])
                # Check if arm is pointing downward (A-pose: Y < -0.15)
                if v_src[1] < -0.15:
                    r_corr = rotation_between(v_src, v_tgt)
                    out_offsets[_key("upperArm", sd)] = r_corr

    def _retarget_spine(self, rot_lh: np.ndarray, rest_rot_lh: np.ndarray, n_frames: int, tracks: dict):
        """Distribute source spine rotation across PMX 上半身 and 上半身2."""
        spine_joints = self.src_map.spine
        tgt_spines = self.tgt_map.spine  # list of PMX bone indices: [上半身, 上半身2, ...]

        if not tgt_spines:
            return

        if spine_joints:
            top_spine = spine_joints[-1]
            p_top = self.source.parents[top_spine]
            r_cur = np.linalg.inv(rot_lh[:, p_top]) @ rot_lh[:, top_spine]
            r_rest = np.linalg.inv(rest_rot_lh[p_top]) @ rest_rot_lh[top_spine]
            total_spine_rot = r_cur @ np.linalg.inv(r_rest)
        else:
            total_spine_rot = np.broadcast_to(np.eye(3), (n_frames, 3, 3))

        if len(tgt_spines) == 1:
            bname = self.target.bones[tgt_spines[0]].name
            tracks[bname] = (np.zeros((n_frames, 3)), mat_to_quat(total_spine_rot))
        else:
            # Distribute 50% / 50% using slerp of quaternion
            q_tot = mat_to_quat(total_spine_rot)
            q_half = Rotation.from_quat(q_tot).as_rotvec() * 0.5
            q_half_quat = Rotation.from_rotvec(q_half).as_quat()

            bname1 = self.target.bones[tgt_spines[0]].name
            bname2 = self.target.bones[tgt_spines[1]].name
            tracks[bname1] = (np.zeros((n_frames, 3)), q_half_quat)
            tracks[bname2] = (np.zeros((n_frames, 3)), q_half_quat)

    def _retarget_arm(
        self,
        sd: str,
        rot_lh: np.ndarray,
        rest_rot_lh: np.ndarray,
        a_pose_offsets: dict,
        n_frames: int,
        tracks: dict,
    ):
        """Retarget Shoulder, UpperArm, LowerArm, Hand with Swing-Twist decomposition."""
        sh_s = self.src_map.get(_key("shoulder", sd))
        ua_s = self.src_map.get(_key("upperArm", sd))
        la_s = self.src_map.get(_key("lowerArm", sd))
        hd_s = self.src_map.get(_key("hand", sd))

        sh_t = self.tgt_map.get(_key("shoulder", sd))
        ua_t = self.tgt_map.get(_key("upperArm", sd))
        la_t = self.tgt_map.get(_key("lowerArm", sd))
        hd_t = self.tgt_map.get(_key("hand", sd))

        # Shoulder
        if sh_s >= 0 and sh_t >= 0:
            p = self.source.parents[sh_s]
            r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, sh_s]
            r_rest = np.linalg.inv(rest_rot_lh[p]) @ rest_rot_lh[sh_s]
            sh_rot = r_cur @ np.linalg.inv(r_rest)
            tracks[self.target.bones[sh_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(sh_rot))

        # Upper Arm with A-pose offset and optional Twist separation
        if ua_s >= 0 and ua_t >= 0:
            p = self.source.parents[ua_s]
            r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, ua_s]
            r_rest = np.linalg.inv(rest_rot_lh[p]) @ rest_rot_lh[ua_s]
            ua_rot = r_cur @ np.linalg.inv(r_rest)

            # Apply A-pose correction if detected
            corr = a_pose_offsets.get(_key("upperArm", sd))
            if corr is not None:
                ua_rot = corr @ ua_rot @ np.linalg.inv(corr)

            arm_twist_t = self.tgt_map.get(_key("armTwist", sd))
            if self.cfg.enable_twist_bones and arm_twist_t >= 0:
                # Decompose rotation along bone axis: X-axis ([1, 0, 0] or [-1, 0, 0])
                axis = np.array([1.0 if sd == "L" else -1.0, 0.0, 0.0])
                q_all = mat_to_quat(ua_rot)
                q_swing, q_twist = swing_twist(q_all, axis)
                tracks[self.target.bones[ua_t].name] = (np.zeros((n_frames, 3)), q_swing)
                tracks[self.target.bones[arm_twist_t].name] = (np.zeros((n_frames, 3)), q_twist)
            else:
                tracks[self.target.bones[ua_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(ua_rot))

        # Lower Arm (Elbow)
        if la_s >= 0 and la_t >= 0:
            p = self.source.parents[la_s]
            r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, la_s]
            r_rest = np.linalg.inv(rest_rot_lh[p]) @ rest_rot_lh[la_s]
            la_rot = r_cur @ np.linalg.inv(r_rest)
            tracks[self.target.bones[la_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(la_rot))

        # Hand (Wrist)
        if hd_s >= 0 and hd_t >= 0:
            p = self.source.parents[hd_s]
            r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, hd_s]
            r_rest = np.linalg.inv(rest_rot_lh[p]) @ rest_rot_lh[hd_s]
            hd_rot = r_cur @ np.linalg.inv(r_rest)

            hand_twist_t = self.tgt_map.get(_key("handTwist", sd))
            if self.cfg.enable_twist_bones and hand_twist_t >= 0:
                axis = np.array([1.0 if sd == "L" else -1.0, 0.0, 0.0])
                q_all = mat_to_quat(hd_rot)
                q_swing, q_twist = swing_twist(q_all, axis)
                tracks[self.target.bones[hd_t].name] = (np.zeros((n_frames, 3)), q_swing)
                tracks[self.target.bones[hand_twist_t].name] = (np.zeros((n_frames, 3)), q_twist)
            else:
                tracks[self.target.bones[hd_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(hd_rot))

    def _retarget_leg(
        self,
        sd: str,
        rot_lh: np.ndarray,
        rest_rot_lh: np.ndarray,
        pos_lh: np.ndarray,
        rest_pos_lh: np.ndarray,
        scale: float,
        n_frames: int,
        tracks: dict,
    ):
        """Retarget Leg: either generate Foot IK / Toe IK or output FK rotations."""
        ul_s = self.src_map.get(_key("upperLeg", sd))
        ll_s = self.src_map.get(_key("lowerLeg", sd))
        ft_s = self.src_map.get(_key("foot", sd))
        to_s = self.src_map.get(_key("toes", sd))

        ul_t = self.tgt_map.get(_key("upperLeg", sd))
        ll_t = self.tgt_map.get(_key("lowerLeg", sd))
        ft_t = self.tgt_map.get(_key("foot", sd))
        to_t = self.tgt_map.get(_key("toes", sd))

        leg_ik_t = self.tgt_map.get(_key("legIK", sd))
        toe_ik_t = self.tgt_map.get(_key("toeIK", sd))

        if self.cfg.enable_foot_ik and leg_ik_t >= 0 and ft_s >= 0:
            # --- Foot IK Generation ---
            # World position of source foot scaled to target stature
            hips_s = self.src_map.get("hips")
            hips_delta = (pos_lh[:, hips_s] - rest_pos_lh[hips_s]) * scale if hips_s >= 0 else 0.0

            # Foot relative to rest hips
            foot_rel = (pos_lh[:, ft_s] - (rest_pos_lh[hips_s] if hips_s >= 0 else 0.0)) * scale
            hips_t = self.tgt_map.get("center")
            hips_t_pos = self.target.bones[hips_t].position if hips_t >= 0 else np.zeros(3)
            tgt_foot_world = hips_t_pos + foot_rel

            # Foot IK position offset relative to Foot IK rest position
            leg_ik_rest = self.target.bones[leg_ik_t].position
            leg_ik_pos = tgt_foot_world - leg_ik_rest

            # Foot rotation from source foot
            p_ft = self.source.parents[ft_s]
            r_cur = np.linalg.inv(rot_lh[:, p_ft]) @ rot_lh[:, ft_s]
            r_rest = np.linalg.inv(rest_rot_lh[p_ft]) @ rest_rot_lh[ft_s]
            ft_rot = r_cur @ np.linalg.inv(r_rest)

            tracks[self.target.bones[leg_ik_t].name] = (leg_ik_pos, mat_to_quat(ft_rot))

            # Toe IK
            if toe_ik_t >= 0 and to_s >= 0:
                toe_rel = (pos_lh[:, to_s] - (rest_pos_lh[hips_s] if hips_s >= 0 else 0.0)) * scale
                tgt_toe_world = hips_t_pos + toe_rel
                toe_ik_rest = self.target.bones[toe_ik_t].position
                toe_ik_pos = tgt_toe_world - toe_ik_rest
                tracks[self.target.bones[toe_ik_t].name] = (
                    toe_ik_pos,
                    mat_to_quat(np.broadcast_to(np.eye(3), (n_frames, 3, 3))),
                )

            # When foot IK is enabled, FK leg bones (足, ひざ) keep zero rotation
            if ul_t >= 0:
                tracks[self.target.bones[ul_t].name] = (
                    np.zeros((n_frames, 3)),
                    mat_to_quat(np.broadcast_to(np.eye(3), (n_frames, 3, 3))),
                )
            if ll_t >= 0:
                tracks[self.target.bones[ll_t].name] = (
                    np.zeros((n_frames, 3)),
                    mat_to_quat(np.broadcast_to(np.eye(3), (n_frames, 3, 3))),
                )
            if ft_t >= 0:
                tracks[self.target.bones[ft_t].name] = (
                    np.zeros((n_frames, 3)),
                    mat_to_quat(np.broadcast_to(np.eye(3), (n_frames, 3, 3))),
                )
        else:
            # --- FK Leg Fallback ---
            if ul_s >= 0 and ul_t >= 0:
                p = self.source.parents[ul_s]
                r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, ul_s]
                r_rest = np.linalg.inv(rest_rot_lh[p]) @ rest_rot_lh[ul_s]
                tracks[self.target.bones[ul_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(r_cur @ np.linalg.inv(r_rest)))

            if ll_s >= 0 and ll_t >= 0:
                p = self.source.parents[ll_s]
                r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, ll_s]
                r_rest = np.linalg.inv(rest_rot_lh[p]) @ rest_rot_lh[ll_s]
                tracks[self.target.bones[ll_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(r_cur @ np.linalg.inv(r_rest)))

            if ft_s >= 0 and ft_t >= 0:
                p = self.source.parents[ft_s]
                r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, ft_s]
                r_rest = np.linalg.inv(rest_rot_lh[p]) @ rest_rot_lh[ft_s]
                tracks[self.target.bones[ft_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(r_cur @ np.linalg.inv(r_rest)))

    def _retarget_fingers(self, rot_lh: np.ndarray, rest_rot_lh: np.ndarray, n_frames: int, tracks: dict):
        """Retarget finger joints (Thumb, Index, Middle, Ring, Little) 1 to 3."""
        for sd in SIDES:
            for f in FINGERS:
                for seg in (1, 2, 3):
                    k = f"{f}{seg}"
                    s_idx = self.src_map.get(_key(k, sd))
                    t_idx = self.tgt_map.get(_key(k, sd))
                    if s_idx >= 0 and t_idx >= 0:
                        p = self.source.parents[s_idx]
                        if p >= 0:
                            r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, s_idx]
                            r_rest = np.linalg.inv(rest_rot_lh[p]) @ rest_rot_lh[s_idx]
                            f_rot = r_cur @ np.linalg.inv(r_rest)
                        else:
                            f_rot = rot_lh[:, s_idx] @ np.linalg.inv(rest_rot_lh[s_idx])
                        bname = self.target.bones[t_idx].name
                        tracks[bname] = (np.zeros((n_frames, 3)), mat_to_quat(f_rot))
