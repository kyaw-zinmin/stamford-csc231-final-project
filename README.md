# stamford-csc231-final-project
Feature-Based Moving Object Tracking System
---
Detects and tracks moving objects from a live webcam feed using:
- Gaussian filtering
- Harris corner detection
- SIFT feature extraction
- Optical flow (Lucas-Kanade)
- HOG shape representation

Interactive Controls:
- q -> quit
- p -> pause/resume
- 1 -> toggle feature points
- 2 -> toggle motion vectors
- 3 -> toggle bounding boxes
- 4 -> toggle trajectories
- h -> toggle corner detector (Harris / Shi-Tomasi)
- m -> toggle mirror display (flip horizontally)
- UP -> increase number of tracked points
- DOWN -> decrease number of tracked points

Output window shows:
  - Tracked feature points (green static, red moving)
  - Motion vectors (cyan arrows)
  - Object bounding boxes (blue rectangles with ID)
  - Object trajectories (yellow trails)
  - Current settings overlay (always readable, not mirrored)

Usage:
  - `python tracker_interactive.py`
    - uses default webcam
  - `python tracker_interactive.py --camera 1`         
    - use camera ID 1
  - `python tracker_interactive.py video.mp4`           
    - process a video file
  - `python tracker_interactive.py video.mp4 --output out.mp4`   
    - save output