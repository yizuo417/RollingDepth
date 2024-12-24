import cv2
import numpy as np


def resize_two_videos_keep_aspect_ratio(video1_path, video2_path, output_path):
    # 打开两个视频文件
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # 获取视频帧率
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    # 获取每个视频的宽度和高度
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 确定输出视频的高度和宽度
    min_height = min(height1, height2)
    min_width1 = int(min_height * width1 / height1)
    min_width2 = int(min_height * width2 / height2)

    # 确定输出视频的帧率
    out_fps = min(fps1, fps2)

    # 输出视频宽度是两个视频的宽度之和
    output_width = min_width1 + min_width2
    output_height = min_height

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (output_width, output_height))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # 按保持长宽比缩放每个视频帧
        frame1 = cv2.resize(frame1, (min_width1, min_height))
        frame2 = cv2.resize(frame2, (min_width2, min_height))

        # 将视频帧拼接在一起
        side_by_side_frame = np.hstack((frame1, frame2))
        out.write(side_by_side_frame)

    # 释放资源
    cap1.release()
    cap2.release()
    out.release()



if __name__ == "__main__":
    video1_path = "/workspace/pyz/RollingDepth/output/iclight/test1222_6/inter_video.mp4"
    video2_path = "/workspace/pyz/RollingDepth/output/iclight/test1222_6/sportman_rgb.mp4"
    #video3_path = ""
    output_path = "compare/iclight_alignment/sportman_sun_sbs_video_1222_6.mp4"

    resize_two_videos_keep_aspect_ratio(video1_path, video2_path,output_path)
