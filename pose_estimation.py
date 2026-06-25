import cv2
import numpy as np

def estimate_pose(ref_path, scene_path):
    # Read images in grayscale
    imgRef = cv2.imread(ref_path, 0)
    imgScene = cv2.cvtColor(scene_path, cv2.COLOR_BGR2GRAY)

    # OpenCV Camera Calibration
    camera_mtx = np.array([
        [4.42625569e+03, 0.00000000e+00, 2.15970939e+03],
        [0.00000000e+00, 4.38970188e+03, 2.87371388e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    camera_dist = np.array([
        [2.56482554e-01, -1.61595387e+00, -1.29318561e-03, -5.99380717e-04, 2.85319202e+00]
    ])
    camera_rvecs = [
        [[-0.05304794], [ 0.08466661], [1.56304828]],
        [[-0.415083  ], [-0.09740391], [1.52731958]],
        [[ 0.18921584], [-0.19645353], [1.5545511 ]],
        [[-0.11073814], [ 0.13010866], [1.55259421]],
        [[ 0.14609498], [ 0.59323426], [1.40571303]],
        [[-0.61036646], [-0.19970287], [1.61382465]],
        [[ 0.14757729], [-0.12499284], [1.54609337]]
    ]
    camera_tvecs = [
        [[ 2.93140911], [-4.70361525], [10.69326393]],
        [[ 3.27257734], [-4.74977468], [12.57755113]],
        [[ 2.71474422], [-2.96708907], [ 8.99299959]],
        [[ 2.61216521], [-2.75794779], [17.25601049]],
        [[ 1.9894337 ], [-3.03919681], [15.96592166]],
        [[ 4.40866103], [-2.48786939], [18.16278295]],
        [[ 2.53563893], [-2.04345809], [11.82366396]]
    ]

    # MATLAB Camera Calibration
    # camera_mtx = np.array([
    #     [4.29625710e+03, 0.00000000e+00, 2.85032370e+03],
    #     [0.00000000e+00, 4.31316960e+03, 2.21138500e+03],
    #     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    # ])
    # camera_dist = np.array([
    #     [3.06000000e-01, -1.32990000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
    # ])
    
    # rotation_vecs = np.array([
    #     [ 9.8000e-03,  8.4370e-01,  5.6000e-03],
    #     [-1.3540e-01,  7.9840e-01, -8.9400e-02],
    #     [ 5.5320e-01, -6.1450e-01, -1.7830e-01],
    #     [ 3.7970e-01, -7.0070e-01, -6.9700e-02],
    #     [ 2.7030e-01, -6.4640e-01, -1.8200e-02],
    #     [-3.5210e-01, -6.9680e-01,  7.2700e-02],
    #     [-2.7580e-01, -6.8930e-01, -5.2200e-02],
    #     [-1.9230e-01, -6.6920e-01, -1.8640e-01],
    #     [-1.2960e-01, -6.5170e-01, -2.6980e-01],
    #     [-1.4660e-01, -2.2300e-02, -7.1500e-02],
    #     [-1.4230e-01, -3.9800e-02, -2.5400e-02],
    #     [ 5.2100e-02,  7.7040e-01,  2.1310e-01],
    #     [ 4.0610e-01, -5.1540e-01,  1.9000e-02],
    #     [-2.2500e-01,  2.5200e-01,  1.7247e+00],
    #     [ 2.9500e-02,  1.3780e-01,  1.1900e-02],
    #     [-1.8600e-02,  9.3960e-01, -4.5000e-03],
    #     [ 1.2540e-01,  9.2010e-01,  1.8050e-01],
    #     [ 2.4200e-02,  7.8990e-01,  5.5800e-02],
    #     [-2.2350e-01,  7.8870e-01, -3.7530e-01],
    # ])

    # translation_vecs = np.array([
    #     [-1.35468e+02, -4.11603e+01,  5.52609e+02],
    #     [-1.53568e+02, -5.62732e+01,  5.56977e+02],
    #     [-7.20286e+01,  1.19130e+01,  3.23659e+02],
    #     [-7.12028e+01, -1.81710e+00,  3.20737e+02],
    #     [-6.75997e+01, -1.71300e+00,  3.07312e+02],
    #     [-2.03066e+01, -8.01410e+01,  3.72826e+02],
    #     [-3.50815e+01, -6.41232e+01,  3.46093e+02],
    #     [-3.81350e+01, -4.62280e+01,  3.22340e+02],
    #     [-5.07159e+01, -3.86469e+01,  3.05914e+02],
    #     [-1.30831e+02, -5.17907e+01,  9.72260e+02],
    #     [-1.31729e+02, -9.49772e+01,  6.93774e+02],
    #     [-1.08352e+02, -9.36705e+01,  5.02558e+02],
    #     [-2.34687e+01,  8.04800e+00,  3.07467e+02],
    #     [ 3.88987e+01, -6.03446e+01,  2.91007e+02],
    #     [-1.25802e+01, -5.12935e+01,  5.37718e+02],
    #     [-8.65909e+01, -5.13726e+01,  5.25772e+02],
    #     [-9.75095e+01, -7.08715e+01,  5.05215e+02],
    #     [-6.77980e+01, -5.77506e+01,  5.51047e+02],
    #     [-2.40110e+01, -1.97941e+01,  5.07722e+02],
    # ])

    term_eps = 1e-6

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
    ptsScene = np.float32([keypointsScene[m.trainIdx] .pt for m in goodMatches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(ptsScene, ptsRef, cv2.RANSAC, 5.0)

    if M is None:
        print("Homography could not be computed.")
        exit()

    ref_h, ref_w = imgRef.shape
    corners = np.float32([[0,0],[0,ref_h-1],[ref_w-1,ref_h-1],[ref_w-1,0]]).reshape(-1,1,2)
    projected_points = cv2.perspectiveTransform(corners, M)

    posterPoints3D = np.array([[0, 0, 0], [0.6096, 0, 0], [0.6096, 0.6096, 0], [0, 0.6096, 0]], dtype=np.float32)

    # Compute camera pose
    if len(projected_points) < 4 or len(projected_points) < 4:
        return
    else:
        # success, rvec, tvec, inliers = cv2.solvePnPRansac(posterPoints3D, projected_points, camera_mtx, camera_dist)
        # success, rvec, tvec, inliers = cv2.solvePnPRansac(posterPoints3D, projected_points, camera_mtx, camera_dist, flags = cv2.SOLVEPNP_IPPE_SQUARE)
        success, rvec, tvec = cv2.solvePnP(posterPoints3D, projected_points, camera_mtx, camera_dist, flags = cv2.SOLVEPNP_IPPE_SQUARE)

    if success:
        rvec, tvec = cv2.solvePnPRefineVVS(posterPoints3D, projected_points, camera_mtx, camera_dist, rvec, tvec, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, term_eps))
    else:
        return "Error: Pose estimation failed.", None

    # Return the results as a dictionary
    return None, {
        "rvec": rvec,
        "tvec": tvec,
        "homography": M,
        "distance_cm": np.linalg.norm(tvec)
    }