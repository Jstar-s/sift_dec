import numpy as np
import cv2


imageA = cv2.imread("left_01.png")
imageB = cv2.imread("right_01.png")
imageA = cv2.resize(imageA, (512, 512))
imageB = cv2.resize(imageB, (512, 512))
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
# 建立sift生成容器
descriptor = cv2.xfeatures2d.SIFT_create()
kpsA, featuresA = descriptor.detectAndCompute(grayA, None)
kpsB, featuresB = descriptor.detectAndCompute(grayB, None)
kpsA = np.float32([kp.pt for kp in kpsA])
kpsB = np.float32([kp.pt for kp in kpsB])
matcher = cv2.BFMatcher()

# 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
# 输出的是特征点的数据结构
matches = []
for m in rawMatches:
    # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        # 存储两个点在featuresA, featuresB中的索引值
        matches.append((m[0].trainIdx, m[0].queryIdx))
# int
# queryIdx – > 是测试图像的特征点描述符（descriptor）的下标，同时也是描述符对应特征点（keypoint)的下标。
# int
# trainIdx – > 是样本图像的特征点描述符的下标，同样也是相应的特征点的下标。
# int
# imgIdx – > 当样本是多张图像的话有用。
# float
# distance – > 代表这一对匹配的特征点描述符（本质是向量）的欧氏距离，数值越小也就说明两个特征点越相像
# 当筛选后的匹配对大于4时，计算视角变换矩阵
if len(matches) > 4:
    # 获取匹配对的点坐标
    ptsA = np.float32([kpsA[i] for (_, i) in matches])
    ptsB = np.float32([kpsB[i] for (i, _) in matches])

# 计算视角变换矩阵
(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 8)
(hA, wA) = imageA.shape[:2]
(hB, wB) = imageB.shape[:2]
vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
vis[0:hA, 0:wA] = imageA
vis[0:hB, wA:] = imageB

# 联合遍历，画出匹配对
for ((trainIdx, queryIdx), s) in zip(matches, status):
    # 当点对匹配成功时，画到可视化图上
    if s == 1:
        # 画出匹配对
        ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
        ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
        print(ptA, end="")
        print(ptB)
        cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        cv2.circle(vis, ptA, 1,(0,0,255))
        cv2.circle(vis, ptB, 1,(255,0,0))

cv2.imshow("Keypoint Matches", vis)
cv2.waitKey(0)

