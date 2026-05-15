"""
landmark_sequencer.py — Topological Landmark-Chain Navigation
=============================================================
Manages an ordered sequence of landmark images that define a navigation path.

How it works:
  1. Load numbered landmark images (00_*, 01_*, 02_*, ... last = final goal)
  2. Encode each landmark into an embedding using GoalEncoder
  3. During flight, compare the current camera view to the ACTIVE landmark
  4. When confidence exceeds `waypoint_threshold`, advance to the next landmark
  5. When the FINAL landmark is confirmed, return action='LAND'

Usage example:
  sequencer = LandmarkSequencer(
      landmark_dir="landmarks/my_mission",
      goal_encoder=goal_enc,
      device=device,
      preprocess=PREPROCESS,
      waypoint_threshold=0.78,   # advance to next waypoint at this confidence
      confirm_frames=4,          # must be confident for N consecutive frames
  )
  result = sequencer.step(current_obs)
  # result: {"action": "MOVE"|"ADVANCE"|"LAND",
  #           "active_idx": int,
  #           "active_name": str,
  #           "goal_confidence": float,
  #           "progress_pct": float}
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Dict

import torch
from PIL import Image
from torchvision import transforms


# ── Helpers ──────────────────────────────────────────────────────────────────

def _natural_sort_key(s: str) -> list:
    """Sort strings containing numbers in human-natural order (00, 01, 10 ...)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


# ── Main class ────────────────────────────────────────────────────────────────

class LandmarkSequencer:
    """
    Manages a chain of landmark images as navigation waypoints.

    Directory layout expected::

        landmarks/my_mission/
            00_start.jpg          ← first waypoint (optional reference)
            01_hallway_corner.jpg ← second waypoint
            02_blue_door.jpg      ← third waypoint
            03_goal_table.jpg     ← final waypoint → triggers LAND

    Landmarks are sorted by filename (natural order), so the numeric prefix
    controls the sequence.  Any image extension supported by PIL is accepted.

    Parameters
    ----------
    landmark_dir : str | Path
        Path to the directory containing landmark images.
    goal_encoder : nn.Module
        The GoalEncoder from the navigation stack.  Must be in eval mode.
    device : torch.device
        Inference device (cpu / cuda / mps).
    preprocess : transforms.Compose
        The same pre-processing pipeline used for live camera frames.
    waypoint_threshold : float
        Similarity confidence above which the current waypoint is considered
        "reached" and the sequencer advances to the next one.
        Recommended range: 0.70–0.85.  Lower than goal_threshold in
        drone_flight.py so waypoints trigger before the final landing check.
    confirm_frames : int
        Number of consecutive frames that must all exceed `waypoint_threshold`
        before advancing.  Guards against single noisy frames (mirrors fix C3).
    goal_threshold : float | None
        Confidence for the FINAL landmark to trigger 'LAND'.  Defaults to
        `waypoint_threshold + 0.05` if not supplied, so the final stop is
        slightly stricter than intermediate waypoints.
    skip_first_landmark : bool
        If True, landmark index 0 is treated as a "start reference" and is
        skipped immediately (the drone starts navigating toward landmark 1).
        Useful when the first photo was taken at the takeoff position.
    """

    _SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        landmark_dir: str | Path,
        goal_encoder,
        device: torch.device,
        preprocess: transforms.Compose,
        waypoint_threshold: float = 0.78,
        confirm_frames: int = 4,
        goal_threshold: Optional[float] = None,
        skip_first_landmark: bool = False,
    ):
        self.landmark_dir       = Path(landmark_dir)
        self.goal_encoder       = goal_encoder
        self.device             = device
        self.preprocess         = preprocess
        self.waypoint_threshold = waypoint_threshold
        self.confirm_frames     = confirm_frames
        self.goal_threshold     = goal_threshold or (waypoint_threshold + 0.05)
        self.skip_first_landmark = skip_first_landmark

        # Internal state
        self._confidence_history: List[float] = []
        self._advance_log: List[Dict]          = []   # for post-flight analysis

        # ── Load and encode all landmark images ──────────────────────────────
        self.landmark_paths: List[Path]  = self._discover_landmarks()
        self.landmark_names: List[str]   = [p.stem for p in self.landmark_paths]
        self.embeddings: List[torch.Tensor] = self._encode_all()

        if len(self.embeddings) == 0:
            raise RuntimeError(
                f"No valid landmark images found in '{self.landmark_dir}'. "
                f"Supported extensions: {self._SUPPORTED_EXTS}"
            )

        # ── Set starting index ────────────────────────────────────────────────
        self.active_idx: int = 1 if (skip_first_landmark and len(self.embeddings) > 1) else 0

        self._print_summary()

    # ── Discovery ─────────────────────────────────────────────────────────────

    def _discover_landmarks(self) -> List[Path]:
        if not self.landmark_dir.exists():
            raise FileNotFoundError(
                f"Landmark directory not found: '{self.landmark_dir}'"
            )
        images = sorted(
            [
                p for p in self.landmark_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self._SUPPORTED_EXTS
            ],
            key=lambda p: _natural_sort_key(p.name),
        )
        return images

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode_single(self, path: Path) -> torch.Tensor:
        img    = Image.open(path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.goal_encoder(tensor)       # (1, D)
        return emb

    def _encode_all(self) -> List[torch.Tensor]:
        embeddings = []
        for i, path in enumerate(self.landmark_paths):
            emb = self._encode_single(path)
            embeddings.append(emb)
            print(f"[Landmarks] Encoded [{i:02d}] {path.name}  → shape {tuple(emb.shape)}")
        return embeddings

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def total_waypoints(self) -> int:
        return len(self.embeddings)

    @property
    def is_final_waypoint(self) -> bool:
        return self.active_idx == len(self.embeddings) - 1

    @property
    def active_embedding(self) -> torch.Tensor:
        return self.embeddings[self.active_idx]

    @property
    def active_name(self) -> str:
        return self.landmark_names[self.active_idx]

    @property
    def progress_pct(self) -> float:
        """Fraction of waypoints completed, 0.0 → 1.0."""
        if len(self.embeddings) <= 1:
            return 0.0
        return self.active_idx / (len(self.embeddings) - 1)

    def get_all_names(self) -> List[str]:
        return list(self.landmark_names)

    def step(self, goal_matcher, current_obs: torch.Tensor) -> Dict:
        """
        Core step: compare current camera embedding to the active landmark.

        Parameters
        ----------
        goal_matcher : GoalMatcher
            The GoalMatcher module from the nav stack.
        current_obs : torch.Tensor
            GoalEncoder output for the current camera frame — shape (1, D).

        Returns
        -------
        dict with keys:
            action          : "MOVE" | "ADVANCE" | "LAND"
            active_idx      : current waypoint index
            active_name     : stem of the active landmark filename
            goal_confidence : similarity score for the active landmark (0–1)
            progress_pct    : fraction of path completed
            waypoints_done  : number of waypoints passed
            total_waypoints : total number of landmarks
        """
        # ── 1. Compute confidence vs active landmark ──────────────────────────
        with torch.no_grad():
            conf = goal_matcher(current_obs, self.active_embedding).item()

        # ── 2. Build confirmation window (mirrors C3 fix) ─────────────────────
        self._confidence_history.append(conf)
        if len(self._confidence_history) > self.confirm_frames:
            self._confidence_history.pop(0)

        window_full     = len(self._confidence_history) == self.confirm_frames
        threshold       = self.goal_threshold if self.is_final_waypoint else self.waypoint_threshold
        window_all_high = window_full and all(c > threshold for c in self._confidence_history)

        # ── 3. Decide action ──────────────────────────────────────────────────
        action = "MOVE"

        if window_all_high:
            if self.is_final_waypoint:
                action = "LAND"
            else:
                action = "ADVANCE"
                self._advance_to_next()

        base_result = {
            "active_idx":      self.active_idx,
            "active_name":     self.active_name,
            "goal_confidence": conf,
            "progress_pct":    self.progress_pct,
            "waypoints_done":  self.active_idx,
            "total_waypoints": self.total_waypoints,
        }

        return {"action": action, **base_result}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _advance_to_next(self) -> None:
        """Move to the next landmark in the chain."""
        prev_idx  = self.active_idx
        prev_name = self.active_name
        self.active_idx += 1
        self._confidence_history.clear()     # fresh window for next waypoint

        log_entry = {
            "from_idx":  prev_idx,
            "from_name": prev_name,
            "to_idx":    self.active_idx,
            "to_name":   self.active_name,
        }
        self._advance_log.append(log_entry)

        print(
            f"[Landmarks] ✅ Waypoint {prev_idx:02d} '{prev_name}' reached  →  "
            f"now targeting [{self.active_idx:02d}] '{self.active_name}'  "
            f"({self.progress_pct*100:.0f}% complete)"
        )

    def _print_summary(self) -> None:
        print("\n[Landmarks] Mission path loaded:")
        print(f"           Directory : {self.landmark_dir}")
        print(f"           Waypoints : {self.total_waypoints}")
        for i, name in enumerate(self.landmark_names):
            marker = "🏁" if i == len(self.landmark_names) - 1 else f"{i:02d}"
            active = " ◄ START" if i == self.active_idx else ""
            print(f"             [{marker}] {name}{active}")
        print(
            f"           Thresholds : waypoint={self.waypoint_threshold:.2f}  "
            f"final={self.goal_threshold:.2f}  "
            f"confirm={self.confirm_frames} frames"
        )
        print()

    def get_advance_log(self) -> List[Dict]:
        """Returns the list of waypoint-advance events for post-flight analysis."""
        return list(self._advance_log)

    def reset(self) -> None:
        """Reset sequencer state (allows re-use for a second mission run)."""
        self.active_idx            = 1 if (self.skip_first_landmark and self.total_waypoints > 1) else 0
        self._confidence_history   = []
        self._advance_log          = []
        print(f"[Landmarks] Sequencer reset → starting at [{self.active_idx:02d}] '{self.active_name}'")
