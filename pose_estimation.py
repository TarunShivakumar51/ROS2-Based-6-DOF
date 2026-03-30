import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimate_pose(ref_path, scene_path):
    # Read images in grayscale
    imgRef = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    imgScene = cv2.cvtColor(scene_path, cv2.COLOR_BGR2GRAY)

    if imgRef is None or imgScene is None:
        return "Error: Could not read one or both images.", None

    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypointsRef, descriptorsRef = sift.detectAndCompute(imgRef, None)
    keypointsScene, descriptorsScene = sift.detectAndCompute(imgScene, None)

    if descriptorsRef is None or descriptorsScene is None:
        return "Error: No features detected.", None

    # Brute-force matcher (L2 norm for SIFT)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptorsRef, descriptorsScene, k=2)

    # Apply ratio test
    goodMatches = [m for m, n in matches if m.distance < 0.5 * n.distance]

    if len(goodMatches) < 4:
        return "Error: Not enough matches to compute homography.", None

    # Get matching keypoint coordinates
    ptsRef = np.float32([keypointsRef[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    ptsScene = np.float32([keypointsScene[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

    # Find homography
    H, mask = cv2.findHomography(ptsRef, ptsScene, cv2.RANSAC, 5.0)

    if H is None:
        return "Error: Homography could not be computed.", None

    hScene, wScene = imgScene.shape
    hRef, wRef = imgRef.shape

    # Project poster corners into the scene
    posterPoints2D = np.array([[0, 0], [wRef, 0], [wRef, hRef], [0, hRef]], dtype=np.float32).reshape(-1, 1, 2)
    projectedPoints = cv2.perspectiveTransform(posterPoints2D, H)

    # Define 3D coordinates (2ft x 2ft = ~61cm)
    posterPoints3D = np.array([[0, 0, 0], [61, 0, 0], [61, 61, 0], [0, 61, 0]], dtype=np.float32)

    # Camera intrinsic matrix (rough estimate)
    focalLength = 1000
    cx, cy = wScene / 2, hScene / 2
    K = np.array([[focalLength, 0, cx], [0, focalLength, cy], [0, 0, 1]])

    # Compute camera pose
    success, rvec, tvec = cv2.solvePnP(posterPoints3D, projectedPoints, K, None)

    if not success:
        return "Error: Pose estimation failed.", None

    # # Optional: Visualization
    # if show_plot:
    #     alignedRef = cv2.warpPerspective(imgRef, H, (wScene, hScene))
    #     overlay = cv2.addWeighted(imgScene, 0.5, alignedRef, 0.5, 0)
    #     plt.imshow(overlay, cmap='gray')
    #     plt.title("Aligned Overlay")
    #     plt.show()

    # Return the results as a dictionary
    return None, {
        "rvec": rvec,
        "tvec": tvec,
        "homography": H,
        "distance_cm": np.linalg.norm(tvec)
    }