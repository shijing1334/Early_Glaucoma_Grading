import os
import pandas as pd


def process_files(input_csv_mapping, file_type='video', file_ext='.mp4', header=False, path_col='video_path',
                  label_col='label'):
    """
    通用的文件处理函数
    :param input_csv_mapping: 输入CSV映射字典 {输入CSV文件名: (对应文件夹名, 输出CSV文件名)}
    :param file_type: 文件类型标识（用于打印日志）
    :param file_ext: 文件扩展名（如.mp4, .jpg）
    :param header: 是否保留CSV表头
    :param path_col: 输出CSV的路径列名
    :param label_col: 输出CSV的标签列名
    """
    # 遍历每个CSV文件及对应的文件夹
    for input_file, (folder_name, output_file) in input_csv_mapping.items():
        # 构建输入CSV的完整路径（在data子文件夹下）
        input_csv_path = os.path.join('data', input_file)

        # 检查输入CSV文件是否存在
        if not os.path.exists(input_csv_path):
            print(f"警告：未找到输入CSV文件 {input_csv_path}，跳过该文件处理")
            continue

        # 读取CSV文件，确保video_name以字符串形式读取
        df_labels = pd.read_csv(input_csv_path, sep=',', dtype={'video_name': str})
        df_labels.columns = df_labels.columns.str.strip()  # 去除列名的空格

        # 打印调试信息
        print(f"\n处理{file_type}文件: {input_csv_path}")
        print("列名:", df_labels.columns.tolist())
        print(df_labels.head())

        # 准备结果列表
        results = []

        # 构建文件夹完整路径（在data子文件夹下）
        folder_path = os.path.join('data', folder_name)

        # 遍历每个条目
        for index, row in df_labels.iterrows():
            file_base_name = row['video_name']
            label = row['label']

            # 构建文件完整路径
            file_full_path = os.path.join(folder_path, f'{file_base_name}{file_ext}')

            # 检查文件是否存在
            if os.path.exists(file_full_path):
                results.append((os.path.abspath(file_full_path), label))
                if file_type == 'image':  # 仅图像文件打印检查路径（保持原有逻辑）
                    print(f"找到{file_type}文件: {file_full_path}")
            else:
                print(f"未找到{file_type}文件: {file_full_path}")

        # 创建结果的DataFrame
        df_results = pd.DataFrame(results, columns=[path_col, label_col])

        # 保存结果到当前目录的CSV文件
        df_results.to_csv(output_file, index=False, header=header)

        print(f"已创建{file_type}CSV文件: {output_file}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 1. 处理视频文件（原有第一段逻辑）
    video_csv_mapping = {
        'trainv.csv': ('train_video', 'train_video.csv'),
        'valv.csv': ('val_video', 'val_video.csv'),
        'testv.csv': ('test_video', 'test_video.csv')
    }
    process_files(
        input_csv_mapping=video_csv_mapping,
        file_type='video',
        file_ext='.mp4',
        header=False,
        path_col='video_path',
        label_col='label'
    )

    # 2. 处理图像文件（原有第二段逻辑）
    image_csv_mapping = {
        'traini.csv': ('train_image', 'train_image.csv'),
        'valv.csv': ('val_image', 'val_image.csv'),
        'testv.csv': ('test_image', 'test_image.csv')
    }
    process_files(
        input_csv_mapping=image_csv_mapping,
        file_type='image',
        file_ext='.jpg',
        header=True,
        path_col='path',
        label_col='class'
    )

    print("\n所有文件处理完成！")