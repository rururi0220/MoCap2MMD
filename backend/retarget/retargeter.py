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
        # 1. Resample source motion to target FPS
        rot, pos = resample_globals(
            self.source.rot, self.source.pos, self.source.fps, self.cfg.target_fps
        )
        n_frames = rot.shape[0]
        scale = self.calculate_scale()

        # 2. Determine reference (base) pose for retargeting
        # Check if frame 0 is a calibration T-pose (arms horizontal), while rest_pos is arms-down
        ua_l = self.src_map.get(_key("upperArm", "L"))
        la_l = self.src_map.get(_key("lowerArm", "L"))
        use_frame0_as_ref = False
        if ua_l >= 0 and la_l >= 0 and n_frames > 0:
            v_rest = self.source.rest_pos[la_l] - self.source.rest_pos[ua_l]
            v_f0 = pos[0, la_l] - pos[0, ua_l]
            if v_rest[1] < -0.1 and abs(v_f0[1]) < 0.15 and abs(v_f0[0]) > 0.05:
                use_frame0_as_ref = True

        if use_frame0_as_ref:
            ref_rot = rot[0]
            ref_pos = pos[0]
        else:
            ref_rot = self.source.rest_rot
            ref_pos = self.source.rest_pos

        # 3. Coordinate conversion: Canonicalize source to Left-Handed MMD (Target)
        # Determine axis alignment with MMD convention:
        # X: Left is +X, Right is -X.
        # Y: Up is +Y.
        # Z: Front is -Z, Back is +Z.
        sx, sy, sz = 1.0, 1.0, 1.0
        # Check lateral axis (Left vs Right)
        lt_s = self.src_map.get(_key("upperLeg", "L"))
        rt_s = self.src_map.get(_key("upperLeg", "R"))
        if lt_s < 0 or rt_s < 0:
            lt_s = self.src_map.get(_key("upperArm", "L"))
            rt_s = self.src_map.get(_key("upperArm", "R"))
        if lt_s >= 0 and rt_s >= 0:
            dx = ref_pos[lt_s, 0] - ref_pos[rt_s, 0]
            if dx < -0.01:
                sx = -1.0  # Left is -X in source; invert to match MMD +X

        # Check vertical axis (Up vs Down)
        hd_s = self.src_map.get("head")
        hp_s = self.src_map.get("hips")
        if hd_s >= 0 and hp_s >= 0:
            dy = ref_pos[hd_s, 1] - ref_pos[hp_s, 1]
            if dy < -0.05:
                sy = -1.0  # Head is below hips; invert Y

        # Check sagittal axis (Front vs Back)
        to_s = self.src_map.get(_key("toes", "L"))
        ft_s = self.src_map.get(_key("foot", "L"))
        if to_s >= 0 and ft_s >= 0:
            dz = ref_pos[to_s, 2] - ref_pos[ft_s, 2]
            if dz > 0.01:
                sz = -1.0  # Toes are in front (+Z in source); invert to match MMD front -Z
        else:
            # Fallback to standard Right-Handed to MMD conversion (flip Z)
            sz = -1.0

        M_conv = np.diag([sx, sy, sz])
        pos_lh = pos * np.array([sx, sy, sz])
        ref_pos_lh = ref_pos * np.array([sx, sy, sz])
        rot_lh = np.einsum("ia,fjab,bk->fjik", M_conv, rot, M_conv)
        ref_rot_lh = np.einsum("ia,jab,bk->jik", M_conv, ref_rot, M_conv)

        tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # bone_name -> (pos_array, rot_quat_array)

        # 4. Rest pose offset analysis (A-pose to T-pose compensation)
        a_pose_offsets: dict[str, np.ndarray] = {}
        if self.cfg.auto_detect_a_pose:
            self._compute_a_pose_offsets(ref_pos_lh, a_pose_offsets)

        # 5. Hips / Center / Root handling
        hips_idx = self.src_map.get("hips")
        center_pmx = self.tgt_map.get("center")

        if hips_idx >= 0 and center_pmx >= 0:
            # Root displacement scaled to PMX stature
            d_pos = (pos_lh[:, hips_idx] - ref_pos_lh[hips_idx]) * scale
            # Hips delta orientation
            hips_delta_rot = rot_lh[:, hips_idx] @ np.linalg.inv(ref_rot_lh[hips_idx])

            center_name = self.target.bones[center_pmx].name
            # In MMD, "センター" is the parent of both "下半身" (pelvis) and "上半身" (spine).
            # It carries both root translation AND overall body orientation.
            tracks[center_name] = (d_pos, mat_to_quat(hips_delta_rot))

            hips_pmx = self.tgt_map.get("hips")
            if hips_pmx >= 0:
                hips_name = self.target.bones[hips_pmx].name
                # 下半身 stays at identity (aligned with センター)
                tracks[hips_name] = (np.zeros((n_frames, 3)), mat_to_quat(np.broadcast_to(np.eye(3), (n_frames, 3, 3))))

        # 6. Spine chain distribution
        self._retarget_spine(rot_lh, ref_rot_lh, n_frames, tracks)

        # 7. Head & Neck
        thorax_s = self.src_map.spine[-1] if self.src_map.spine else self.src_map.get("hips")
        neck_s = self.src_map.get("neck")
        head_s = self.src_map.get("head")

        neck_t = self.tgt_map.get("neck")
        head_t = self.tgt_map.get("head")

        if neck_s >= 0 and neck_t >= 0 and thorax_s >= 0:
            rel_cur = np.linalg.inv(rot_lh[:, thorax_s]) @ rot_lh[:, neck_s]
            rel_ref = np.linalg.inv(ref_rot_lh[thorax_s]) @ ref_rot_lh[neck_s]
            neck_rot = rel_cur @ np.linalg.inv(rel_ref)
            tracks[self.target.bones[neck_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(neck_rot))

        if head_s >= 0 and head_t >= 0:
            parent_s = neck_s if neck_s >= 0 else thorax_s
            if parent_s >= 0:
                rel_cur = np.linalg.inv(rot_lh[:, parent_s]) @ rot_lh[:, head_s]
                rel_ref = np.linalg.inv(ref_rot_lh[parent_s]) @ ref_rot_lh[head_s]
                head_rot = rel_cur @ np.linalg.inv(rel_ref)
            else:
                head_rot = rot_lh[:, head_s] @ np.linalg.inv(ref_rot_lh[head_s])
            tracks[self.target.bones[head_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(head_rot))

        # 8. Arms and Limbs
        for sd in SIDES:
            self._retarget_arm(sd, rot_lh, ref_rot_lh, pos_lh, ref_pos_lh, a_pose_offsets, n_frames, tracks)
            self._retarget_leg(sd, rot_lh, ref_rot_lh, pos_lh, ref_pos_lh, scale, n_frames, tracks)

        # 9. Fingers
        if self.cfg.enable_fingers:
            self._retarget_fingers(rot_lh, ref_rot_lh, n_frames, tracks)

        # 10. Build VMDMotion object
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

    def _compute_a_pose_offsets(self, ref_pos_lh: np.ndarray, out_offsets: dict[str, np.ndarray]):
        """Detect A-pose in target PMX model vs source reference pose and compute compensation."""
        for sd in SIDES:
            ua_s = self.src_map.get(_key("upperArm", sd))
            la_s = self.src_map.get(_key("lowerArm", sd))
            ua_t = self.tgt_map.get(_key("upperArm", sd))
            la_t = self.tgt_map.get(_key("lowerArm", sd))
            if ua_s >= 0 and la_s >= 0 and ua_t >= 0 and la_t >= 0:
                p_ua_t = self.target.bones[ua_t].position
                p_la_t = self.target.bones[la_t].position
                v_tgt = normalize(p_la_t - p_ua_t)
                v_src = normalize(ref_pos_lh[la_s] - ref_pos_lh[ua_s])
                if np.linalg.norm(v_tgt - v_src) > 0.05:
                    r_corr = rotation_between(v_tgt, v_src)
                    out_offsets[_key("upperArm", sd)] = r_corr

    def _retarget_spine(self, rot_lh: np.ndarray, ref_rot_lh: np.ndarray, n_frames: int, tracks: dict):
        """Distribute source spine rotation across PMX 上半身 and 上半身2."""
        hips_s = self.src_map.get("hips")
        spine_joints = self.src_map.spine
        tgt_spines = self.tgt_map.spine

        if not tgt_spines:
            return

        top_s = spine_joints[-1] if spine_joints else -1

        if top_s >= 0 and hips_s >= 0:
            rel_cur = np.linalg.inv(rot_lh[:, hips_s]) @ rot_lh[:, top_s]
            rel_ref = np.linalg.inv(ref_rot_lh[hips_s]) @ ref_rot_lh[top_s]
            total_spine_rot = rel_cur @ np.linalg.inv(rel_ref)
        elif top_s >= 0:
            p_top = self.source.parents[top_s]
            rel_cur = np.linalg.inv(rot_lh[:, p_top]) @ rot_lh[:, top_s]
            rel_ref = np.linalg.inv(ref_rot_lh[p_top]) @ ref_rot_lh[top_s]
            total_spine_rot = rel_cur @ np.linalg.inv(rel_ref)
        else:
            total_spine_rot = np.broadcast_to(np.eye(3), (n_frames, 3, 3))

        if len(tgt_spines) == 1:
            bname = self.target.bones[tgt_spines[0]].name
            tracks[bname] = (np.zeros((n_frames, 3)), mat_to_quat(total_spine_rot))
        else:
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
        ref_rot_lh: np.ndarray,
        pos_lh: np.ndarray,
        ref_pos_lh: np.ndarray,
        a_pose_offsets: dict,
        n_frames: int,
        tracks: dict,
    ):
        """Retarget Shoulder, UpperArm, LowerArm, Hand using 3D joint direction vectors."""
        sh_s = self.src_map.get(_key("shoulder", sd))
        ua_s = self.src_map.get(_key("upperArm", sd))
        la_s = self.src_map.get(_key("lowerArm", sd))
        hd_s = self.src_map.get(_key("hand", sd))

        sh_t = self.tgt_map.get(_key("shoulder", sd))
        ua_t = self.tgt_map.get(_key("upperArm", sd))
        la_t = self.tgt_map.get(_key("lowerArm", sd))
        hd_t = self.tgt_map.get(_key("hand", sd))

        # Rest vectors of target PMX model
        if ua_t >= 0 and la_t >= 0:
            v_ua_rest = normalize(self.target.bones[la_t].position - self.target.bones[ua_t].position)
        else:
            v_ua_rest = np.array([1.0 if sd == "L" else -1.0, 0.0, 0.0])

        if la_t >= 0 and hd_t >= 0:
            v_la_rest = normalize(self.target.bones[hd_t].position - self.target.bones[la_t].position)
        else:
            v_la_rest = v_ua_rest

        # Shoulder
        if sh_s >= 0 and sh_t >= 0:
            p = self.source.parents[sh_s]
            r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, sh_s]
            r_ref = np.linalg.inv(ref_rot_lh[p]) @ ref_rot_lh[sh_s]
            tracks[self.target.bones[sh_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(r_cur @ np.linalg.inv(r_ref)))

        # Upper Arm
        if ua_s >= 0 and ua_t >= 0:
            p = self.source.parents[ua_s]
            r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, ua_s]
            r_ref = np.linalg.inv(ref_rot_lh[p]) @ ref_rot_lh[ua_s]
            ua_rot = r_cur @ np.linalg.inv(r_ref)

            # Apply A-pose correction if detected
            corr = a_pose_offsets.get(_key("upperArm", sd))
            if corr is not None:
                ua_rot = ua_rot @ corr

            tracks[self.target.bones[ua_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(ua_rot))

        # Lower Arm (Elbow)
        if la_s >= 0 and la_t >= 0:
            p_la = self.source.parents[la_s]
            r_la_cur = np.linalg.inv(rot_lh[:, p_la]) @ rot_lh[:, la_s]
            r_la_ref = np.linalg.inv(ref_rot_lh[p_la]) @ ref_rot_lh[la_s]
            la_rot = r_la_cur @ np.linalg.inv(r_la_ref)
            tracks[self.target.bones[la_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(la_rot))

        # Hand (Wrist)
        if hd_s >= 0 and hd_t >= 0:
            p_hd = self.source.parents[hd_s]
            if p_hd >= 0:
                r_hd_cur = np.linalg.inv(rot_lh[:, p_hd]) @ rot_lh[:, hd_s]
                r_hd_ref = np.linalg.inv(ref_rot_lh[p_hd]) @ ref_rot_lh[hd_s]
                hd_rot = r_hd_cur @ np.linalg.inv(r_hd_ref)
            else:
                hd_rot = np.broadcast_to(np.eye(3), (n_frames, 3, 3))
            tracks[self.target.bones[hd_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(hd_rot))

        # Lock twist bones to identity so they don't break the arm hierarchy
        arm_twist_t = self.tgt_map.get(_key("armTwist", sd))
        if arm_twist_t >= 0:
            tracks[self.target.bones[arm_twist_t].name] = (
                np.zeros((n_frames, 3)),
                mat_to_quat(np.broadcast_to(np.eye(3), (n_frames, 3, 3))),
            )
        hand_twist_t = self.tgt_map.get(_key("handTwist", sd))
        if hand_twist_t >= 0:
            tracks[self.target.bones[hand_twist_t].name] = (
                np.zeros((n_frames, 3)),
                mat_to_quat(np.broadcast_to(np.eye(3), (n_frames, 3, 3))),
            )

    def _retarget_leg(
        self,
        sd: str,
        rot_lh: np.ndarray,
        ref_rot_lh: np.ndarray,
        pos_lh: np.ndarray,
        ref_pos_lh: np.ndarray,
        scale: float,
        n_frames: int,
        tracks: dict,
    ):
        """Retarget Leg: anatomical FK bone orientations and FK-derived Foot IK target."""
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

        if ul_s >= 0 and ll_s >= 0 and ft_s >= 0 and ul_t >= 0 and ll_t >= 0 and ft_t >= 0:
            # 1. Thigh relative rotation from pelvis
            p_thigh = self.source.parents[ul_s]
            r_cur = np.linalg.inv(rot_lh[:, p_thigh]) @ rot_lh[:, ul_s]
            r_ref = np.linalg.inv(ref_rot_lh[p_thigh]) @ ref_rot_lh[ul_s]
            r_thigh = r_cur @ np.linalg.inv(r_ref)

            # 2. Knee relative rotation from thigh
            p_knee = self.source.parents[ll_s]
            r_cur = np.linalg.inv(rot_lh[:, p_knee]) @ rot_lh[:, ll_s]
            r_ref = np.linalg.inv(ref_rot_lh[p_knee]) @ ref_rot_lh[ll_s]
            r_knee = r_cur @ np.linalg.inv(r_ref)

            # 3. Foot relative rotation from knee
            p_foot = self.source.parents[ft_s]
            r_cur = np.linalg.inv(rot_lh[:, p_foot]) @ rot_lh[:, ft_s]
            r_ref = np.linalg.inv(ref_rot_lh[p_foot]) @ ref_rot_lh[ft_s]
            r_foot = r_cur @ np.linalg.inv(r_ref)

            # Always populate FK leg tracks with anatomical relative rotations
            tracks[self.target.bones[ul_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(r_thigh))
            tracks[self.target.bones[ll_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(r_knee))
            tracks[self.target.bones[ft_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(r_foot))

            # 4. Foot IK target placed at the exact FK foot reach
            if self.cfg.enable_foot_ik and leg_ik_t >= 0:
                p_ul = self.target.bones[ul_t].position
                p_ll = self.target.bones[ll_t].position
                p_ft = self.target.bones[ft_t].position
                p_ik = self.target.bones[leg_ik_t].position

                v_thigh_rest = p_ll - p_ul
                v_shin_rest = p_ft - p_ll

                v_thigh = np.einsum("fij,j->fi", r_thigh, v_thigh_rest)
                r_knee_world = r_thigh @ r_knee
                v_shin = np.einsum("fij,j->fi", r_knee_world, v_shin_rest)
                r_foot_rel = v_thigh + v_shin

                center_pmx = self.tgt_map.get("center")
                center_name = self.target.bones[center_pmx].name if center_pmx >= 0 else None
                R_center = quat_to_mat(tracks[center_name][1]) if center_name and center_name in tracks else np.broadcast_to(np.eye(3), (n_frames, 3, 3))
                p_center_rest = self.target.bones[center_pmx].position if center_pmx >= 0 else np.zeros(3)
                c_pos = tracks[center_name][0] if center_name and center_name in tracks else np.zeros((n_frames, 3))

                # Hip world position
                p_hip_world = p_center_rest + c_pos + np.einsum("fij,j->fi", R_center, (p_ul - p_center_rest))

                # Natural FK foot target in world space
                p_ik_world = p_hip_world + np.einsum("fij,fj->fi", R_center, r_foot_rel)
                leg_ik_pos = p_ik_world - p_ik

                # Foot world orientation with zero sideways roll
                ft_rot = rot_lh[:, ft_s] @ np.linalg.inv(ref_rot_lh[ft_s])
                v_fwd = ft_rot @ np.array([0.0, 0.0, -1.0])
                yaw = np.arctan2(-v_fwd[:, 0], -v_fwd[:, 2])
                pitch = np.arcsin(np.clip(-v_fwd[:, 1], -1.0, 1.0))
                eulers = np.stack([yaw, pitch, np.zeros_like(yaw)], axis=-1)
                ft_quat = Rotation.from_euler("yxz", eulers, degrees=False).as_quat()

                tracks[self.target.bones[leg_ik_t].name] = (leg_ik_pos, ft_quat)

                if toe_ik_t >= 0:
                    tracks[self.target.bones[toe_ik_t].name] = (
                        np.zeros((n_frames, 3)),
                        mat_to_quat(np.broadcast_to(np.eye(3), (n_frames, 3, 3))),
                    )


    def _retarget_fingers(self, rot_lh: np.ndarray, ref_rot_lh: np.ndarray, n_frames: int, tracks: dict):
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
                            r_ref = np.linalg.inv(ref_rot_lh[p]) @ ref_rot_lh[s_idx]
                            f_rot = r_cur @ np.linalg.inv(r_ref)
                        else:
                            f_rot = rot_lh[:, s_idx] @ np.linalg.inv(ref_rot_lh[s_idx])
                        bname = self.target.bones[t_idx].name
                        tracks[bname] = (np.zeros((n_frames, 3)), mat_to_quat(f_rot))
