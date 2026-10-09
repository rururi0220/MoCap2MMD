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

        # Detect inverted (Y-down) skeleton (common in optical/markerless mocap like Theia3D)
        head_or_top = self.src_map.get("head")
        if head_or_top < 0:
            head_or_top = self.src_map.get("neck")
        hips_idx = self.src_map.get("hips")
        if head_or_top >= 0 and hips_idx >= 0:
            up_vec = self.source.rest_pos[head_or_top] - self.source.rest_pos[hips_idx]
            if up_vec[1] < -0.1:
                # Apply 180-deg rotation around X axis: [x, y, z] -> [x, -y, -z] to keep right-handedness
                RX_180 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
                self.source.rest_pos = self.source.rest_pos @ RX_180.T
                self.source.pos = self.source.pos @ RX_180.T
                self.source.rest_rot = np.einsum("ab,jbc->jac", RX_180, self.source.rest_rot)
                self.source.rot = np.einsum("ab,fjbc->fjac", RX_180, self.source.rot)

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
            if v_rest[1] < -0.1 and abs(v_f0[1]) < 0.15 and v_f0[0] > 0.05:
                use_frame0_as_ref = True

        if use_frame0_as_ref:
            ref_rot = rot[0]
            ref_pos = pos[0]
        else:
            ref_rot = self.source.rest_rot
            ref_pos = self.source.rest_pos

        # 3. Coordinate conversion: Right-Handed (Source) to Left-Handed MMD (Target)
        # R_lh = Mz @ R_rh @ Mz
        # pos_lh = pos_rh * [1, 1, -1]
        rot_lh = np.einsum("ia,fjab,bk->fjik", MIRROR_Z, rot, MIRROR_Z)
        pos_lh = pos * np.array([1.0, 1.0, -1.0])
        ref_rot_lh = np.einsum("ia,jab,bk->jik", MIRROR_Z, ref_rot, MIRROR_Z)
        ref_pos_lh = ref_pos * np.array([1.0, 1.0, -1.0])

        tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # bone_name -> (pos_array, rot_quat_array)

        # 4. Rest pose offset analysis (A-pose to T-pose compensation if reference pose is in A-pose)
        a_pose_offsets: dict[str, np.ndarray] = {}
        if self.cfg.auto_detect_a_pose and not use_frame0_as_ref:
            self._compute_a_pose_offsets(ref_pos_lh, a_pose_offsets)

        # 5. Hips / Center / Root handling
        hips_idx = self.src_map.get("hips")
        center_pmx = self.tgt_map.get("center")
        groove_pmx = self.tgt_map.get("groove")

        if hips_idx >= 0 and center_pmx >= 0:
            # Root displacement scaled to PMX stature
            d_pos = (pos_lh[:, hips_idx] - ref_pos_lh[hips_idx]) * scale
            # Hips delta orientation
            hips_delta_rot = rot_lh[:, hips_idx] @ np.linalg.inv(ref_rot_lh[hips_idx])

            center_name = self.target.bones[center_pmx].name
            # In MMD, "センター" usually carries position + rotation, or position only if "下半身" takes rotation
            hips_pmx = self.tgt_map.get("hips")
            if hips_pmx >= 0:
                tracks[center_name] = (d_pos, mat_to_quat(np.broadcast_to(np.eye(3), (n_frames, 3, 3))))
                hips_name = self.target.bones[hips_pmx].name
                tracks[hips_name] = (np.zeros((n_frames, 3)), mat_to_quat(hips_delta_rot))
            else:
                tracks[center_name] = (d_pos, mat_to_quat(hips_delta_rot))

        # 6. Spine chain distribution
        self._retarget_spine(rot_lh, ref_rot_lh, n_frames, tracks)

        # 7. Head & Neck
        for sem in ("neck", "head"):
            s_idx = self.src_map.get(sem)
            t_idx = self.tgt_map.get(sem)
            if s_idx >= 0 and t_idx >= 0:
                p_s = self.source.parents[s_idx]
                if p_s >= 0:
                    r_cur = np.linalg.inv(rot_lh[:, p_s]) @ rot_lh[:, s_idx]
                    r_ref = np.linalg.inv(ref_rot_lh[p_s]) @ ref_rot_lh[s_idx]
                    local_r = r_cur @ np.linalg.inv(r_ref)
                else:
                    local_r = rot_lh[:, s_idx] @ np.linalg.inv(ref_rot_lh[s_idx])
                bname = self.target.bones[t_idx].name
                tracks[bname] = (np.zeros((n_frames, 3)), mat_to_quat(local_r))

        # 8. Arms and Limbs
        for sd in SIDES:
            self._retarget_arm(sd, rot_lh, ref_rot_lh, a_pose_offsets, n_frames, tracks)
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
        """Detect A-pose in upper arms and compute compensation to T-pose (horizontal)."""
        for sd in SIDES:
            ua = self.src_map.get(_key("upperArm", sd))
            la = self.src_map.get(_key("lowerArm", sd))
            if ua >= 0 and la >= 0:
                v_src = normalize(ref_pos_lh[la] - ref_pos_lh[ua])
                # Target T-pose arm vector in MMD: Left arm goes +X [1, 0, 0], Right arm goes -X [-1, 0, 0]
                v_tgt = np.array([1.0 if sd == "L" else -1.0, 0.0, 0.0])
                # Check if arm is pointing downward (A-pose: Y < -0.15)
                if v_src[1] < -0.15:
                    # Rotation to orient MMD horizontal arm down to source rest arm direction
                    r_corr = rotation_between(v_tgt, v_src)
                    out_offsets[_key("upperArm", sd)] = r_corr

    def _retarget_spine(self, rot_lh: np.ndarray, ref_rot_lh: np.ndarray, n_frames: int, tracks: dict):
        """Distribute source spine rotation across PMX 上半身 and 上半身2."""
        spine_joints = self.src_map.spine
        tgt_spines = self.tgt_map.spine  # list of PMX bone indices: [上半身, 上半身2, ...]

        if not tgt_spines:
            return

        if spine_joints:
            top_spine = spine_joints[-1]
            p_top = self.source.parents[top_spine]
            r_cur = np.linalg.inv(rot_lh[:, p_top]) @ rot_lh[:, top_spine]
            r_ref = np.linalg.inv(ref_rot_lh[p_top]) @ ref_rot_lh[top_spine]
            total_spine_rot = r_cur @ np.linalg.inv(r_ref)
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
        ref_rot_lh: np.ndarray,
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
            r_ref = np.linalg.inv(ref_rot_lh[p]) @ ref_rot_lh[sh_s]
            sh_rot = r_cur @ np.linalg.inv(r_ref)
            tracks[self.target.bones[sh_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(sh_rot))

        # Upper Arm with A-pose offset and optional Twist separation
        if ua_s >= 0 and ua_t >= 0:
            p = self.source.parents[ua_s]
            r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, ua_s]
            r_ref = np.linalg.inv(ref_rot_lh[p]) @ ref_rot_lh[ua_s]
            ua_rot = r_cur @ np.linalg.inv(r_ref)

            # Apply A-pose correction if detected
            corr = a_pose_offsets.get(_key("upperArm", sd))
            if corr is not None:
                ua_rot = ua_rot @ corr

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
            r_ref = np.linalg.inv(ref_rot_lh[p]) @ ref_rot_lh[la_s]
            la_rot = r_cur @ np.linalg.inv(r_ref)
            tracks[self.target.bones[la_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(la_rot))

        # Hand (Wrist)
        if hd_s >= 0 and hd_t >= 0:
            p = self.source.parents[hd_s]
            r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, hd_s]
            r_ref = np.linalg.inv(ref_rot_lh[p]) @ ref_rot_lh[hd_s]
            hd_rot = r_cur @ np.linalg.inv(r_ref)

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
        ref_rot_lh: np.ndarray,
        pos_lh: np.ndarray,
        ref_pos_lh: np.ndarray,
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
            # Foot displacement from reference standing pose scaled to target stature
            leg_ik_pos = (pos_lh[:, ft_s] - ref_pos_lh[ft_s]) * scale

            # Foot rotation from source foot
            p_ft = self.source.parents[ft_s]
            r_cur = np.linalg.inv(rot_lh[:, p_ft]) @ rot_lh[:, ft_s]
            r_ref = np.linalg.inv(ref_rot_lh[p_ft]) @ ref_rot_lh[ft_s]
            ft_rot = r_cur @ np.linalg.inv(r_ref)

            tracks[self.target.bones[leg_ik_t].name] = (leg_ik_pos, mat_to_quat(ft_rot))

            # Toe IK
            if toe_ik_t >= 0 and to_s >= 0:
                toe_ik_pos = (pos_lh[:, to_s] - ref_pos_lh[to_s]) * scale
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
                r_ref = np.linalg.inv(ref_rot_lh[p]) @ ref_rot_lh[ul_s]
                tracks[self.target.bones[ul_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(r_cur @ np.linalg.inv(r_ref)))

            if ll_s >= 0 and ll_t >= 0:
                p = self.source.parents[ll_s]
                r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, ll_s]
                r_ref = np.linalg.inv(ref_rot_lh[p]) @ ref_rot_lh[ll_s]
                tracks[self.target.bones[ll_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(r_cur @ np.linalg.inv(r_ref)))

            if ft_s >= 0 and ft_t >= 0:
                p = self.source.parents[ft_s]
                r_cur = np.linalg.inv(rot_lh[:, p]) @ rot_lh[:, ft_s]
                r_ref = np.linalg.inv(ref_rot_lh[p]) @ ref_rot_lh[ft_s]
                tracks[self.target.bones[ft_t].name] = (np.zeros((n_frames, 3)), mat_to_quat(r_cur @ np.linalg.inv(r_ref)))

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
