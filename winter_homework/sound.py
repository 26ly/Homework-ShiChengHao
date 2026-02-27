import numpy as np
import librosa
import scipy.signal as signal
import warnings
from pydub import AudioSegment

warnings.filterwarnings('ignore')

class AudioFileHitDetector:
    """
    音频文件命中检测器
    处理传入的音频文件，检测小球命中装甲板的声音并计数
    """

    def __init__(self, sample_rate=44100):
        """
        初始化检测器
        
        Args:
            sample_rate: 目标采样率 (Hz)
        """
        self.sample_rate = sample_rate
        self.hit_count = 0
        
        # 信号处理参数
        self.frame_length = 1024  # 帧长(约23ms)
        self.hop_length = 512     # 帧移(50%重叠)
        
        # 带通滤波器参数 (装甲板命中声的主要频率范围)
        self.lowcut = 800    # Hz
        self.highcut = 5000  # Hz
        
        # 检测阈值
        self.energy_threshold_multiplier = 5.0  # 能量阈值倍数
        self.zcr_threshold = 0.15               # 过零率阈值
        self.spectral_centroid_min = 1500       # Hz
        self.spectral_centroid_max = 3500       # Hz
        self.min_hit_interval = 0.08            # 最小命中间隔(秒)
        
        # 检测结果存储
        self.hit_timestamps = []  # 命中时间戳列表
        
    def load_audio_file(self, file_path, duration=None):
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            duration: 读取时长(秒)，None表示读取全部
            
        Returns:
            audio: 音频信号数组
            sr: 实际采样率
        """
        import os
        from pathlib import Path
        import tempfile
        import atexit

        print(f"正在加载音频文件: {file_path}")

        # 存在性检查
        p = Path(file_path)
        if not p.exists():
            alt = Path(str(file_path).replace('\\', '/'))
            if alt.exists():
                p = alt
            else:
                return None, None
        
        file_to_load = str(p)
        temp_wav_file = None

        # 如果不是wav文件，尝试用pydub转换为wav
        if p.suffix.lower() != '.wav':
            print(f"检测到非WAV文件 ({p.suffix})，尝试转换为WAV格式...")
            try:
                audio_segment = AudioSegment.from_file(file_to_load)
                
                # 创建一个临时WAV文件
                temp_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
                os.close(temp_fd)

                print(f"  正在导出到临时WAV文件: {temp_wav_path}")
                audio_segment.export(temp_wav_path, format="wav")
                
                file_to_load = temp_wav_path
                temp_wav_file = temp_wav_path

                # 注册一个函数，在程序退出时删除临时文件
                def cleanup_temp_file(path):
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                            print(f"已清理临时文件: {path}")
                        except OSError as e:
                            print(f"清理临时文件失败: {e}")
                
                atexit.register(cleanup_temp_file, temp_wav_file)

            except Exception as e:
                print(f"使用pydub转换文件失败: {e}")
                print("请确保已安装FFmpeg并将其添加至系统PATH环境变量中。")
                print("FFmpeg是处理M4A等非WAV格式文件所必需的。")
                return None, None

        try:
            audio, sr = librosa.load(file_to_load, sr=self.sample_rate, duration=duration)
            print(f"加载成功: 时长={len(audio)/sr:.2f}秒, 采样率={sr}Hz")
            return audio, sr
        except Exception as e:
            return None, None
    
    def preprocess_audio(self, audio):
        """
        预处理音频：带通滤波 + 简单降噪
        Args:
            audio: 原始音频信号
            
        Returns:
            处理后的音频
        """
        if len(audio) == 0:
            return audio
        
        print("开始音频预处理...")
        
        # 1. 归一化
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio_norm = audio / max_amp
        else:
            audio_norm = audio
        
        # 2. 带通滤波 (保留命中声的关键频率)
        print(f"  带通滤波: {self.lowcut}-{self.highcut}Hz")
        nyquist = 0.5 * self.sample_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        audio_filtered = signal.filtfilt(b, a, audio_norm)
        
        # 3. 简单降噪 (谱减法简化版)
        print("  降噪处理...")
        if len(audio_filtered) > self.frame_length:
            # 分帧处理
            frames = librosa.util.frame(audio_filtered, 
                                       frame_length=self.frame_length,
                                       hop_length=self.hop_length)
            
            # 计算每帧能量
            frame_energies = np.mean(frames**2, axis=0)
            
            # 估计噪声水平 (前10%的帧作为噪声参考)
            noise_frames = max(1, int(len(frame_energies) * 0.1))
            noise_level = np.mean(frame_energies[:noise_frames])
            
            # 简单的能量门限
            threshold = noise_level * 2
            audio_clean = audio_filtered.copy()
            
            # 对低能量区域进行衰减
            for i in range(frames.shape[1]):
                if frame_energies[i] < threshold:
                    start = i * self.hop_length
                    end = min(start + self.frame_length, len(audio_clean))
                    audio_clean[start:end] *= 0.3  # 衰减低能量区域
        else:
            audio_clean = audio_filtered
        
        print("音频预处理完成")
        return audio_clean
    
    def extract_features(self, audio):
        """
        提取音频特征用于检测
        
        Args:
            audio: 预处理后的音频
            
        Returns:
            features_list: 特征字典列表
            timestamps: 时间戳列表
        """
        
        print("提取音频特征...")
        
        # 计算总帧数
        n_frames = (len(audio) - self.frame_length) // self.hop_length + 1
        timestamps = np.arange(n_frames) * self.hop_length / self.sample_rate

        # 批量特征计算
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            center=False
        )[0]
        energies = rms ** 2

        zcrs = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            center=False
        )[0]

        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=1024,
            hop_length=self.hop_length,
            center=False
        )[0]

        # 冲击特征：上升时间（ms）
        frames = librosa.util.frame(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        envelope = np.abs(signal.hilbert(frames, axis=0))
        attack_time_ms = np.argmax(envelope, axis=0) / self.sample_rate * 1000

        # 统一所有特征数组的长度
        min_len = min(len(energies), len(zcrs), len(spectral_centroids), len(attack_time_ms))
        energies = energies[:min_len]
        zcrs = zcrs[:min_len]
        spectral_centroids = spectral_centroids[:min_len]
        attack_time_ms = attack_time_ms[:min_len]
        timestamps = timestamps[:min_len]

        features = {
            'energy': energies,
            'zcr': zcrs,
            'spectral_centroid': spectral_centroids,
            'attack_time': attack_time_ms
        }

        print(f"特征提取完成: {n_frames}帧")
        return features, timestamps.tolist()
    
    def detect_hits(self, features, timestamps):
        """
        基于特征检测命中事件
        
        Args:
            features: 特征字典（批量特征）
            timestamps: 时间戳列表
            
        Returns:
            命中时间戳列表
        """
        if not features:
            return []
        
        print("开始命中检测...")
        
        # 提取特征数组
        energies = features['energy']
        zcrs = features['zcr']
        spectral_centroids = features['spectral_centroid']
        attack_times = features['attack_time']
        
        # 自适应能量阈值
        background_energy = np.percentile(energies, 30)  # 30%分位数作为背景
        energy_threshold = background_energy * self.energy_threshold_multiplier
        
        print(f"  背景能量: {background_energy:.6f}")
        print(f"  能量阈值: {energy_threshold:.6f}")
        
        # 检测命中
        # 检测条件
        energy_condition = energies > energy_threshold
        zcr_condition = zcrs < self.zcr_threshold
        spectral_condition = (
            (spectral_centroids > self.spectral_centroid_min) &
            (spectral_centroids < self.spectral_centroid_max)
        )
        attack_condition = attack_times < 15  # 上升时间小于15ms

        is_hit = energy_condition & zcr_condition & spectral_condition & attack_condition
        hit_indices = np.where(is_hit)[0].tolist()
        
        # 后处理：合并相邻检测，避免重复计数
        processed_hits = self._post_process_hits(hit_indices, timestamps, features)
        
        print(f"检测完成: 发现{len(processed_hits)}次命中")
        return processed_hits
    
    def _post_process_hits(self, hit_indices, timestamps, features):
        """
        后处理：合并相邻检测，并选择能量峰值点作为命中时刻
        
        Args:
            hit_indices: 检测到的命中索引列表
            timestamps: 时间戳列表
            features: 包含能量信息的特征字典
            
        Returns:
            处理后的命中时间戳列表
        """
        if not hit_indices:
            return []
        
        energies = features['energy']
        
        # 按时间间隔合并
        merged_hits = []
        current_group = [hit_indices[0]]
        
        for i in range(1, len(hit_indices)):
            # 检查时间间隔
            time_diff = timestamps[hit_indices[i]] - timestamps[hit_indices[i-1]]
            
            if time_diff < self.min_hit_interval:
                # 时间间隔太小，合并到同一组
                current_group.append(hit_indices[i])
            else:
                # 新的一组开始，处理上一组
                # 在组内寻找能量最高的点
                peak_energy_index_in_group = np.argmax([energies[j] for j in current_group])
                true_hit_index = current_group[peak_energy_index_in_group]
                merged_hits.append(timestamps[true_hit_index])
                
                # 开始新的一组
                current_group = [hit_indices[i]]
        
        # 处理最后一组
        if current_group:
            peak_energy_index_in_group = np.argmax([energies[j] for j in current_group])
            true_hit_index = current_group[peak_energy_index_in_group]
            merged_hits.append(timestamps[true_hit_index])
        
        return merged_hits
    
    def process_audio_file(self, file_path, duration=None):
        """
        处理音频文件的完整流程
        
        Args:
            file_path: 音频文件路径
            duration: 读取时长(秒)
            
        Returns:
            命中次数
        """
        print("=" * 60)
        print("开始处理音频文件")
        print("=" * 60)
        
        # 重置计数器
        self.hit_count = 0
        self.hit_timestamps = []
        
        # 1. 加载音频文件
        audio, sr = self.load_audio_file(file_path, duration)
        if audio is None:
            print("无法加载音频文件，处理终止")
            return 0
        
        # 2. 预处理
        processed_audio = self.preprocess_audio(audio)
        
        # 3. 提取特征
        features, timestamps = self.extract_features(processed_audio)
        
        # 4. 检测命中
        hit_times = self.detect_hits(features, timestamps)
        
        # 5. 更新结果
        self.hit_count = len(hit_times)
        self.hit_timestamps = hit_times
        
        # 6. 输出结果
        self._print_results(len(audio)/sr, hit_times)
        
        return self.hit_count
    
    def _print_results(self, audio_duration, hit_times):
        """
        打印检测结果
        
        Args:
            audio_duration: 音频时长(秒)
            hit_times: 命中时间戳列表
        """
        print("\n" + "=" * 60)
        print("检测结果")
        print("=" * 60)
        print(f"音频时长: {audio_duration:.2f}秒")
        print(f"命中次数: {self.hit_count}")
        
        if self.hit_count > 0:
            print("\n命中时间点:")
            for i, hit_time in enumerate(hit_times, 1):
                print(f"  命中 #{i}: {hit_time:.3f}秒")
            
            # 统计信息
            if len(hit_times) > 1:
                intervals = np.diff(hit_times)
                print(f"\n统计信息:")
                print(f"  平均命中间隔: {np.mean(intervals):.3f}秒")
                print(f"  最小命中间隔: {np.min(intervals):.3f}秒")
                print(f"  最大命中间隔: {np.max(intervals):.3f}秒")
        
        print("=" * 60)
    
    def get_detection_summary(self):
        """
        获取检测摘要
        
        Returns:
            摘要字典
        """
        return {
            'hit_count': self.hit_count,
            'hit_timestamps': self.hit_timestamps,
            'detection_params': {
                'energy_threshold_multiplier': self.energy_threshold_multiplier,
                'zcr_threshold': self.zcr_threshold,
                'spectral_centroid_range': [self.spectral_centroid_min, self.spectral_centroid_max],
                'min_hit_interval': self.min_hit_interval
            }
        }
    
    def adjust_parameters(self, energy_mult=None, zcr_thresh=None, 
                         centroid_min=None, centroid_max=None, min_interval=None):
        """
        调整检测参数
        
        Args:
            energy_mult: 能量阈值倍数
            zcr_thresh: 过零率阈值
            centroid_min: 谱质心最小值
            centroid_max: 谱质心最大值
            min_interval: 最小命中间隔
        """
        if energy_mult is not None:
            self.energy_threshold_multiplier = energy_mult
            print(f"能量阈值倍数调整为: {energy_mult}")
        
        if zcr_thresh is not None:
            self.zcr_threshold = zcr_thresh
            print(f"过零率阈值调整为: {zcr_thresh}")
        
        if centroid_min is not None:
            self.spectral_centroid_min = centroid_min
            print(f"谱质心最小值调整为: {centroid_min}Hz")
        
        if centroid_max is not None:
            self.spectral_centroid_max = centroid_max
            print(f"谱质心最大值调整为: {centroid_max}Hz")
        
        if min_interval is not None:
            self.min_hit_interval = min_interval
            print(f"最小命中间隔调整为: {min_interval}秒")


# ==================== 使用示例 ====================

def main():
    """主函数示例"""
    print("音频文件命中检测系统")
    print("=" * 60)
    
    # 创建检测器
    detector = AudioFileHitDetector(sample_rate=44100)
    print("D:/1.m4a")
    
    example_file_path = "D:/1.m4a"
    
    # 调用：
    hit_count = detector.process_audio_file(example_file_path)
    
        
   # 获取检测摘要
    summary = detector.get_detection_summary()
        
    print(f"\n检测摘要:")
    print(f"  命中次数: {summary['hit_count']}")
    print(f"  命中时间点: {summary['hit_timestamps']}")
        
   # 参数调整示例
    print("\n" + "=" * 60)
    print("参数调整示例")
    print("=" * 60)
        
   # 调整参数（更敏感的设置）
    detector.adjust_parameters(
        energy_mult=2.0,      # 降低能量阈值，更敏感
        zcr_thresh=0.12,      # 降低过零率阈值
         min_interval=0.037     # 减小最小间隔以适应快速连击
        )
        
    # 重新处理
    print("\n使用新参数重新处理...")
    detector.process_audio_file(example_file_path)
    
    return detector


# ==================== 简单的文件处理函数 ====================

def process_single_file(file_path, sample_rate=44100):
    """
    简单的单文件处理函数
    
    Args:
        file_path: 音频文件路径
        sample_rate: 采样率
        
    Returns:
        命中次数
    """
    detector = AudioFileHitDetector(sample_rate=sample_rate)
    hit_count = detector.process_audio_file(file_path)
    return hit_count, detector.get_detection_summary()


def process_multiple_files(file_paths, sample_rate=44100):
    """
    批量处理多个音频文件
    
    Args:
        file_paths: 音频文件路径列表
        sample_rate: 采样率
        
    Returns:
        结果字典
    """
    results = {}
    
    for file_path in file_paths:
        print(f"\n处理文件: {file_path}")
        detector = AudioFileHitDetector(sample_rate=sample_rate)
        hit_count = detector.process_audio_file(file_path)
        
        results[file_path] = {
            'hit_count': hit_count,
            'hit_timestamps': detector.hit_timestamps,
            'detector': detector
        }
    
    return results


# ==================== 直接运行 ====================

if __name__ == "__main__":
    # 运行主函数
    main()