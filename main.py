"""
Live Webcam Feature-Based Moving Object Tracking System
--------------------------------------------------------
Detects and tracks moving objects from a live webcam feed using:
- Gaussian filtering
- Harris corner detection
- SIFT feature extraction
- Optical flow (Lucas-Kanade)
- HOG shape representation

Interactive Controls:
    q       quit
    p       pause/resume
    1       toggle feature points
    2       toggle motion vectors
    3       toggle bounding boxes
    4       toggle trajectories
    h       toggle corner detector (Harris / Shi-Tomasi)
    m       toggle mirror display (flip horizontally)
    UP      increase number of tracked points
    DOWN    decrease number of tracked points

Output window shows:
  - Tracked feature points (green static, red moving)
  - Motion vectors (cyan arrows)
  - Object bounding boxes (blue rectangles with ID)
  - Object trajectories (yellow trails)
  - Current settings overlay (always readable, not mirrored)

Usage:
    python tracker_interactive.py                     # uses default webcam
    python tracker_interactive.py --camera 1          # use camera ID 1
    python tracker_interactive.py video.mp4           # process a video file
    python tracker_interactive.py video.mp4 --output out.mp4   # save output
"""

import cv2
import numpy as np
import sys
import argparse
from collections import deque

# ---------------------- Parameters ----------------------
MOTION_THRESHOLD = 2.0            # Minimum displacement (pixels) to consider a point moving
GROUP_DISTANCE = 30.0              # Max distance between points to belong to the same object
MIN_GROUP_SIZE = 5                 # Minimum number of points to form an object
MAX_TRAJECTORY_LEN = 30             # Number of past centroids to keep for trajectory
NEW_FEATURE_INTERVAL = 15           # Re-detect corners every N frames (0 to disable)
MAX_POINTS_LIMIT = 500              # Upper limit for point count

LK_PARAMS = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# HOG parameters (for 64x128 detection window, commonly used for people)
HOG_DESCRIPTOR = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)

# ---------------------- Helper Functions ----------------------
def clamp_bbox(bbox, shape):
    """Ensure bounding box (x,y,w,h) lies within the image."""
    x, y, w, h = bbox
    x = max(0, min(x, shape[1] - 1))
    y = max(0, min(y, shape[0] - 1))
    w = min(w, shape[1] - x)
    h = min(h, shape[0] - y)
    return (x, y, w, h)

def compute_hog_for_patch(patch):
    """Resize patch to 64x128 and compute HOG features. Return feature vector or None."""
    if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
        return None
    try:
        resized = cv2.resize(patch, (64, 128), interpolation=cv2.INTER_LINEAR)
        hog = HOG_DESCRIPTOR.compute(resized)
        return hog.flatten()
    except Exception:
        return None

def draw_arrow(img, pt1, pt2, color, thickness=1):
    """Draw arrow from pt1 to pt2."""
    cv2.arrowedLine(img, (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])), color, thickness, tipLength=0.3)

# ---------------------- Main Tracker Class ----------------------
class MovingObjectTracker:
    def __init__(self, source=0, output_path=None):
        """
        source: either a video file path or an integer camera index.
        """
        if isinstance(source, str):
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video file {source}")
        else:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open camera index {source}")

        self.output_path = output_path
        self.writer = None
        if output_path:
            # Get frame properties from the capture source
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:  # webcam may return 0, set a default
                fps = 30.0
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Tracking state
        self.tracks = []                # list of current points [[x,y], ...]
        self.prev_gray = None            # previous grayscale frame
        self.frame_idx = 0
        self.next_object_id = 0
        self.objects = {}                # id -> TrackedObject

        # Interactive controls
        self.show_points = True
        self.show_vectors = True
        self.show_boxes = True
        self.show_trajectories = True
        self.use_harris = False           # default to Shi-Tomasi (goodFeaturesToTrack)
        self.max_features = 100
        self.mirror_display = False       # flip output horizontally
        self.force_redetect = False

        # For re-detection
        self.harris_threshold = 0.01      # initial threshold, can be adjusted

    def detect_features(self, gray):
        """
        Detect corner points using either Harris or Shi-Tomasi.
        Returns array of shape (N,1,2) with float32 points.
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

        if self.use_harris:
            # Harris corner detection
            harris = cv2.cornerHarris(blurred, blockSize=2, ksize=3, k=0.04)
            # Get top self.max_features corners by response
            harris_flat = harris.flatten()
            if len(harris_flat) > self.max_features:
                # Use partition to get indices of top responses
                indices = np.argpartition(harris_flat, -self.max_features)[-self.max_features:]
                rows, cols = np.unravel_index(indices, harris.shape)
                corners = np.column_stack((cols, rows)).astype(np.float32)  # (x,y)
            else:
                # Take all corners above a very low threshold
                y_coords, x_coords = np.where(harris > 1e-5)
                corners = np.column_stack((x_coords, y_coords)).astype(np.float32)

            if len(corners) < MIN_GROUP_SIZE:
                # Fallback to Shi-Tomasi if too few corners
                corners = cv2.goodFeaturesToTrack(blurred, maxCorners=self.max_features,
                                                  qualityLevel=0.01, minDistance=10)
                if corners is not None:
                    corners = corners.squeeze()
                else:
                    corners = np.array([])
        else:
            # Shi-Tomasi (goodFeaturesToTrack)
            corners = cv2.goodFeaturesToTrack(blurred, maxCorners=self.max_features,
                                              qualityLevel=0.01, minDistance=10)
            if corners is not None:
                corners = corners.squeeze()
            else:
                corners = np.array([])

        # Convert to required shape (N,1,2) float32
        if len(corners) == 0:
            return np.array([])
        return corners.reshape(-1, 1, 2).astype(np.float32)

    def process_frame(self, frame):
        """
        Main processing for one frame.
        Returns an image with all visual elements (points, vectors, boxes, trajectories)
        but WITHOUT the status overlay. Overlay is added later in run() after mirroring.
        """
        self.frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.0)   # Gaussian filtering

        # Handle forced re-detection (e.g., after changing settings)
        if self.force_redetect or self.frame_idx == 1:
            self.tracks = self.detect_features(gray)
            self.force_redetect = False
            self.prev_gray = gray
            return frame.copy()   # no overlay yet

        if len(self.tracks) == 0:
            # No points to track – re-detect
            self.tracks = self.detect_features(gray)
            self.prev_gray = gray
            return frame.copy()

        # Optical flow (Lucas-Kanade)
        new_tracks, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray,
                                                            self.tracks, None, **LK_PARAMS)

        # Keep only successful tracks
        good_new = new_tracks[status.flatten() == 1]
        good_old = self.tracks[status.flatten() == 1]

        if len(good_new) == 0:
            # All points lost – reinitialize
            self.tracks = self.detect_features(gray)
            self.prev_gray = gray
            return frame.copy()

        # Compute displacement
        displacement = np.linalg.norm(good_new - good_old, axis=2).flatten()

        # Classify moving points
        moving_mask = displacement > MOTION_THRESHOLD

        # Group moving points into objects (simple distance-based clustering)
        moving_points = good_new[moving_mask].reshape(-1, 2)
        moving_indices = np.where(moving_mask)[0]

        # Prepare for drawing: we will overlay on a copy of the frame
        display = frame.copy()

        # Draw motion vectors and points based on toggles
        if self.show_vectors or self.show_points:
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                if self.show_points:
                    if moving_mask[i]:
                        color = (0, 0, 255)      # red for moving points
                    else:
                        color = (0, 255, 0)      # green for static points
                    cv2.circle(display, (int(new[0,0]), int(new[0,1])), 3, color, -1)
                if self.show_vectors:
                    draw_arrow(display, old[0], new[0], (255, 255, 0), 1)   # cyan arrow

        # If we have moving points, group them into objects
        objects_this_frame = []   # list of (points indices, centroid, bbox)
        if len(moving_points) >= MIN_GROUP_SIZE:
            # Simple greedy clustering
            used = set()
            for i, pt in enumerate(moving_points):
                if i in used:
                    continue
                # Start a new cluster
                cluster_indices = [i]
                cluster_points = [pt]
                used.add(i)
                # Find nearby points
                for j, other in enumerate(moving_points):
                    if j in used:
                        continue
                    if np.linalg.norm(pt - other) < GROUP_DISTANCE:
                        cluster_indices.append(j)
                        cluster_points.append(other)
                        used.add(j)
                if len(cluster_points) >= MIN_GROUP_SIZE:
                    # Compute centroid and bounding box
                    pts_array = np.array(cluster_points)
                    centroid = np.mean(pts_array, axis=0)
                    min_xy = np.min(pts_array, axis=0)
                    max_xy = np.max(pts_array, axis=0)
                    bbox = (int(min_xy[0]), int(min_xy[1]),
                            int(max_xy[0] - min_xy[0]), int(max_xy[1] - min_xy[1]))
                    bbox = clamp_bbox(bbox, gray.shape)
                    objects_this_frame.append((cluster_indices, centroid, bbox))

        # Update persistent objects
        self.update_objects(objects_this_frame, gray, frame)

        # Draw objects and trajectories based on toggles
        if self.show_boxes or self.show_trajectories:
            for obj_id, obj in self.objects.items():
                if self.show_boxes:
                    # Draw bounding box
                    x, y, w, h = obj.bbox
                    cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(display, f"ID:{obj_id}", (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                if self.show_trajectories:
                    # Draw trajectory
                    for i in range(1, len(obj.centroids)):
                        if obj.centroids[i-1] is not None and obj.centroids[i] is not None:
                            cv2.line(display,
                                     (int(obj.centroids[i-1][0]), int(obj.centroids[i-1][1])),
                                     (int(obj.centroids[i][0]), int(obj.centroids[i][1])),
                                     (0, 255, 255), 2)   # yellow

        # Re-detect new features periodically to replace lost ones
        if NEW_FEATURE_INTERVAL > 0 and self.frame_idx % NEW_FEATURE_INTERVAL == 0:
            new_features = self.detect_features(gray)
            if len(new_features) > 0:
                # Merge with existing tracks (simple: add new points if not too close)
                existing = self.tracks.reshape(-1, 2)
                for pt in new_features:
                    pt_flat = pt.flatten()
                    if np.min(np.linalg.norm(existing - pt_flat, axis=1)) > GROUP_DISTANCE:
                        self.tracks = np.vstack([self.tracks, pt.reshape(1,1,2)])

        # Update tracks for next frame
        self.tracks = good_new
        self.prev_gray = gray

        return display   # no status overlay yet

    def _draw_status(self, img):
        """Draw current settings on the image (always in readable orientation)."""
        lines = [
            f"Points: {'ON' if self.show_points else 'OFF'} (1)",
            f"Vectors: {'ON' if self.show_vectors else 'OFF'} (2)",
            f"Boxes: {'ON' if self.show_boxes else 'OFF'} (3)",
            f"Traj: {'ON' if self.show_trajectories else 'OFF'} (4)",
            f"Corner: {'Harris' if self.use_harris else 'Shi-Tomasi'} (h)",
            f"Mirror: {'ON' if self.mirror_display else 'OFF'} (m)",
            f"Max Points: {self.max_features} (UP/DOWN)"
        ]
        y0, dy = 30, 25
        for i, line in enumerate(lines):
            y = y0 + i*dy
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,0,0), 1, cv2.LINE_AA)

    def update_objects(self, objects_this_frame, gray, color_frame):
        """Match detected groups to existing objects and update state."""
        # For simplicity, match by nearest centroid
        new_objects = {}
        used_obj_ids = set()
        for cluster_indices, centroid, bbox in objects_this_frame:
            best_id = None
            best_dist = float('inf')
            for obj_id, obj in self.objects.items():
                if obj_id in used_obj_ids:
                    continue
                last_centroid = obj.centroids[-1]
                if last_centroid is None:
                    continue
                dist = np.linalg.norm(centroid - last_centroid)
                if dist < GROUP_DISTANCE * 1.5 and dist < best_dist:
                    best_dist = dist
                    best_id = obj_id
            if best_id is not None:
                # Update existing object
                obj = self.objects[best_id]
                obj.centroids.append(centroid)
                obj.bbox = bbox
                used_obj_ids.add(best_id)
                new_objects[best_id] = obj

                # Compute HOG for the object patch (for shape representation)
                patch = color_frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                hog_vec = compute_hog_for_patch(patch)
                if hog_vec is not None:
                    obj.hog_features.append(hog_vec)   # store last few if needed
            else:
                # Create new object
                obj_id = self.next_object_id
                self.next_object_id += 1
                obj = TrackedObject(obj_id, centroid, bbox)
                obj.centroids.append(centroid)
                new_objects[obj_id] = obj
        self.objects = new_objects

    def run(self):
        pause = False
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of stream or cannot read frame.")
                break

            if not pause:
                display = self.process_frame(frame)
            else:
                display = frame.copy()   # show raw frame when paused

            # Apply mirroring if enabled (only to the video content)
            if self.mirror_display:
                display = cv2.flip(display, 1)

            # Draw status overlay AFTER mirroring so text is always readable
            self._draw_status(display)

            cv2.imshow('Moving Object Tracker - Interactive', display)
            if self.writer and not pause:
                self.writer.write(display)

            # Get key using waitKeyEx to support arrow keys
            key = cv2.waitKeyEx(1)

            # Regular ASCII keys (mask with 0xFF)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('p'):
                pause = not pause
            elif key & 0xFF == ord('1'):
                self.show_points = not self.show_points
            elif key & 0xFF == ord('2'):
                self.show_vectors = not self.show_vectors
            elif key & 0xFF == ord('3'):
                self.show_boxes = not self.show_boxes
            elif key & 0xFF == ord('4'):
                self.show_trajectories = not self.show_trajectories
            elif key & 0xFF == ord('h'):
                self.use_harris = not self.use_harris
                self.force_redetect = True
            elif key & 0xFF == ord('m'):
                self.mirror_display = not self.mirror_display

            # Arrow keys: common codes (0x26=38, 0x28=40 from waitKey; 63232,63233 from waitKeyEx)
            if key == 82 or key == 0x26 or key == 63232:   # Up arrow
                self.max_features = min(self.max_features + 10, MAX_POINTS_LIMIT)
                self.force_redetect = True
            elif key == 84 or key == 0x28 or key == 63233: # Down arrow
                self.max_features = max(self.max_features - 10, 10)
                self.force_redetect = True

        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

class TrackedObject:
    """Simple structure to hold object state."""
    def __init__(self, obj_id, centroid, bbox):
        self.id = obj_id
        self.centroids = deque(maxlen=MAX_TRAJECTORY_LEN)
        self.centroids.append(centroid)
        self.bbox = bbox
        self.hog_features = []   # list of HOG vectors (optional)

# ---------------------- Main ----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live webcam feature-based moving object tracking with interactive controls')
    parser.add_argument('source', nargs='?', default=0,
                        help='Video file path or camera index (default: 0 for webcam)')
    parser.add_argument('--camera', type=int, dest='camera_id',
                        help='Specify camera ID explicitly (overrides source if given)')
    parser.add_argument('--output', help='Path to output video file (optional)')
    args = parser.parse_args()

    # Determine source: if --camera is given, use that; else interpret source as file path or int.
    if args.camera_id is not None:
        source = args.camera_id
    else:
        # Try to convert source to int if possible (for webcam), otherwise keep as string
        try:
            source = int(args.source)
        except ValueError:
            source = args.source

    tracker = MovingObjectTracker(source, args.output)
    tracker.run()