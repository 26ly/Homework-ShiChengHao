import cv2
import numpy as np

# --- 请按顺序在图像上点击下面四个赛场点对应的像素（顺序必须与 field_points 一致） ---
field_points = np.array([
    [17.45877, -4.22194, 0.4],   # 敌方左前哨站
    [9.535,  -8.93591,  0.0],# 右矿区
    [9.535,   -8.3591,  0.0],    # 左矿区
    [18.91038,-10.8,    0.8]     # 敌方英雄高地
], dtype=np.float64)

camera_matrix = np.array([
    [800.0,   0.0, 320.0],
    [0.0,   800.0, 240.0],
    [0.0,     0.0,   1.0]
], dtype=np.float64)

dist_coeffs = np.zeros((5, 1), dtype=np.float64)

img = cv2.imread("赛场.png")
if img is None:
    print("无法加载图像: 赛场.png，确认路径和文件名。")
    exit(1)

window_name = "Image"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

clicked_points = []
pose_computed = False
rvec = None
tvec = None

def draw_points(im, pts):
    for i, (x, y) in enumerate(pts):
        cv2.circle(im, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(im, str(i+1), (int(x)+6, int(y)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

def mouse_callback(event, x, y, flags, param):
    global clicked_points, pose_computed, rvec, tvec
    if event == cv2.EVENT_LBUTTONUP:
        clicked_points.append((x, y))
        print("鼠标位置：({}, {})".format(x, y))

        # 前 4 次点击用于对应 field_points -> image points
        if len(clicked_points) == 4 and not pose_computed:
            image_points = np.array(clicked_points[:4], dtype=np.float64)
            object_points = field_points.astype(np.float64)

            retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not retval:
                print("solvePnP 失败。检查点对应是否正确、相机内参是否准确。")
            else:
                pose_computed = True
                R, _ = cv2.Rodrigues(rvec)
                print("旋转向量(rvec):", rvec.ravel())
                print("平移向量(tvec):", tvec.ravel())
                print("旋转矩阵 R:\n", R)
                rt = np.hstack((R, tvec.reshape(3,1)))
                matrix_44 = np.vstack((rt, [0,0,0,1]))
                print("4x4 变换矩阵 (相机->场景):\n", matrix_44)

        # 第 5 次点击视为堡垒中心（已知相机坐标系下 z=7.9）
        elif len(clicked_points) >= 5 and pose_computed:
            ux, uy = clicked_points[4]
            uv1 = np.array([ux, uy, 1.0], dtype=np.float64)

            # 已知相机坐标系的z
            z_cam = 7.9

            # 计算相机坐标系下的三维点 Xc = z * K^{-1} * [u,v,1]^T
            invK = np.linalg.inv(camera_matrix)
            Xc = z_cam * (invK.dot(uv1))  # shape (3,)

            # 把相机坐标转换到世界（赛场）坐标
            R, _ = cv2.Rodrigues(rvec)    # world->camera: Xc = R * Xw + t
            t = tvec.reshape(3)
            Xw = R.T.dot(Xc - t)         # Xw = R^T * (Xc - t)

            print("相机坐标系下 (x, y, z):", Xc.tolist())
            print("赛场坐标 (x, y, z):", Xw.tolist())

cv2.setMouseCallback(window_name, mouse_callback)

print("使用说明：按顺序点击图像上与下面 field_points 对应的四个已知点，随后点击堡垒中心（假设 z=0）。")
while True:
    disp = img.copy()
    draw_points(disp, clicked_points)
    cv2.imshow(window_name, disp)
    key = cv2.waitKey(10) & 0xFF
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        print("窗口已关闭，退出。")
        break

cv2.destroyAllWindows()
