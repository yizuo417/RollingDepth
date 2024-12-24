import cv2
import numpy as np


def resize_videos(video1_path, video2_path, video3_path,video4_path, output_path):
    # 打开三个视频文件
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    cap3 = cv2.VideoCapture(video3_path)
    cap4 = cv2.VideoCapture(video4_path)

    # 获取视频1的帧率、宽度和高度
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 获取视频2的帧率、宽度和高度
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 获取视频3的帧率、宽度和高度
    fps3 = cap3.get(cv2.CAP_PROP_FPS)
    width3 = int(cap3.get(cv2.CAP_PROP_FRAME_WIDTH))
    height3 = int(cap3.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 获取视频3的帧率、宽度和高度
    fps4 = cap4.get(cv2.CAP_PROP_FPS)
    width4 = int(cap4.get(cv2.CAP_PROP_FRAME_WIDTH))
    height4 = int(cap4.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 确定最小高度和最小宽度
    min_height = min(height1, height2, height3,height4)
    min_width = min(width1, width2, width3,width4)

    # 创建一个新的视频写入对象，宽度是三个最小宽度之和，高度是最小高度
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = min(fps1, fps2, fps3,fps4)
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (min_width * 4, min_height))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()

        if not ret1 or not ret2 or not ret3 or not ret4:
            break

        # 调整每个视频帧的大小
        frame1 = cv2.resize(frame1, (min_width, min_height))
        frame2 = cv2.resize(frame2, (min_width, min_height))
        frame3 = cv2.resize(frame3, (min_width, min_height))
        frame4 = cv2.resize(frame4, (min_width, min_height))

        # 将三个帧并排组合
        side_by_side_frame = np.hstack((frame1, frame2, frame3,frame4))
        out.write(side_by_side_frame)

    # 释放资源
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    out.release()


if __name__ == "__main__":
    video1_path = "/workspace/pyz/RollingDepth/data/iclight/fg/sportman.mp4"
    video2_path = "/workspace/pyz/RollingDepth/data/iclight/6_sportman_sun.mp4"
    video3_path = "/workspace/pyz/RollingDepth/output/test12_fast_my/1_sportman_sun_rgb.mp4"
    video4_path = "/workspace/pyz/RollingDepth/output/iclight/test1222_7/sportman_rgb.mp4"
    output_path = "compare/sportman_sun_sbs_video(orignal+iclight+without1+with1).mp4"

    resize_videos(video1_path, video2_path, video3_path,video4_path, output_path)
