"""
Simplified Action Classifier for CCTV System
Uses basic heuristics instead of pose detection for activity recognition.
"""

import cv2
import numpy as np
import logging
from collections import defaultdict, deque
from typing import Dict, Optional

log = logging.getLogger(__name__)


class ActionClassifier:
    """Simplified action classifier using bounding box movement patterns."""

    def __init__(self):
        self.track_history = defaultdict(lambda: deque(maxlen=10))  # Track position history
        self.track_actions = {}  # Current actions for each track
        self.frame_count = 0

    def detect_pose(self, frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        """Dummy method for compatibility - returns None since we don't use pose detection."""
        return None

    def _classify_action(self, landmarks: np.ndarray) -> str:
        """Dummy method for compatibility - not used in simplified version."""
        return "unknown"

    def update_frame(self, frame: np.ndarray, boxes: Optional[np.ndarray],
                    track_ids: Optional[np.ndarray]) -> Dict[int, str]:
        """
        Update action classification for current frame using movement patterns.

        Args:
            frame: Input frame (not used in simplified version)
            boxes: array of bounding boxes [x1, y1, x2, y2]
            track_ids: array of tracking IDs

        Returns:
            dict mapping person_id -> action string
        """
        self.frame_count += 1
        actions = {}

        if boxes is None or len(boxes) == 0:
            return actions

        # Update track history
        for i, box in enumerate(boxes):
            try:
                person_id = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else i

                # Extract bbox info
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                # Store (center_x, center_y, width, height, frame)
                self.track_history[person_id].append((center_x, center_y, width, height, self.frame_count))

            except Exception as e:
                log.debug(f"Error updating history for box {i}: {e}")

        # Classify actions for active tracks
        for person_id, history in self.track_history.items():
            if len(history) >= 3:  # Need at least 3 frames for movement analysis
                action = self._classify_movement_action(person_id, history)
                actions[person_id] = action
                self.track_actions[person_id] = action

        # Clean up old tracks (not seen for 30 frames)
        active_person_ids = set()
        if track_ids is not None:
            active_person_ids = set(int(track_ids[i]) if i < len(track_ids) else i for i in range(len(boxes)))

        to_remove = []
        for person_id in self.track_history:
            if person_id not in active_person_ids:
                # Check if last seen more than 30 frames ago
                if history and self.frame_count - history[-1][4] > 30:
                    to_remove.append(person_id)

        for person_id in to_remove:
            del self.track_history[person_id]
            if person_id in self.track_actions:
                del self.track_actions[person_id]

        return actions

    def _classify_movement_action(self, person_id, history):
        """Classify action based on movement patterns."""
        if len(history) < 3:
            return "standing"

        # Get recent positions (last 3 frames)
        recent = list(history)[-3:]

        # Calculate movement vectors
        movements = []
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            movements.append((dx, dy))

        # Calculate average movement
        avg_dx = np.mean([m[0] for m in movements])
        avg_dy = np.mean([m[1] for m in movements])

        # Calculate movement magnitude
        movement_magnitude = np.sqrt(avg_dx**2 + avg_dy**2)

        # Calculate size changes (for potential fighting detection)
        sizes = [h[2] * h[3] for h in recent]  # width * height
        size_change = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0

        # Classify based on movement patterns
        if movement_magnitude < 2:  # Very little movement
            return "standing"
        elif movement_magnitude < 10:  # Slow movement
            return "walking"
        elif size_change > 0.1:  # Significant size changes (potential fighting)
            return "fighting"
        else:  # Moderate to fast movement
            return "running"

    def get_action(self, person_id):
        """Get current action for a track."""
        return self.track_actions.get(person_id, "unknown")

    def release(self):
        """Clean up resources."""
        self.track_history.clear()
        self.track_actions.clear()
