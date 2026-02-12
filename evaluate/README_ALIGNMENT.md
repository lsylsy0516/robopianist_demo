# MIDI Alignment Evaluation

## 概述

`evaluate_alignment.py` 通过比较**视频音频**和**参考MIDI文件**评估机器人演奏表现。结合动态时间规整（DTW）和音频相似度（余弦相似度）两种方法计算综合评分。

**核心思路**：将视频中的机器人演奏音频与预期演奏（参考MIDI合成音频）进行对齐和比较。

## 使用方法

### 基本用法

```bash
# 标准评估（默认启用音频相似度）
python evaluate_alignment.py <video_path> <performed_midi_path>

# 仅使用DTW对齐评分（禁用音频相似度）
python evaluate_alignment.py <video_path> <performed_midi_path> --no-audio
```

### 示例

```bash
# 评估机器人演奏（视频+MIDI）
python evaluate_alignment.py 00002.mp4 twinkle-twinkle-trimmed.mid

# 指定自定义参考MIDI文件
python evaluate_alignment.py 00002.mp4 twinkle-twinkle-trimmed.mid --reference-midi custom_reference.mid

# 调整音频相似度权重（默认0.2）
python evaluate_alignment.py 00002.mp4 twinkle-twinkle-trimmed.mid --audio-weight 0.3

# 仅使用DTW对齐评分
python evaluate_alignment.py 00002.mp4 twinkle-twinkle-trimmed.mid --no-audio
```

## 评分机制

评分范围：0.0（最差）到 1.0（最好）

### 评分指标

#### DTW对齐评分（基础评分）

1. **音高匹配** (权重 100)：严重惩罚音高错误
2. **时间精度** (权重 1)：开始时间差异
3. **时长匹配** (权重 1)：音符时长差异
4. **完整性**：缺失音符惩罚 500/个，多余音符惩罚 200/个

#### 音频相似度评分（可选）

启用 `--use-audio` 后，计算：
- 从视频提取音频波形（使用 `librosa`）
- 从MIDI合成音频（优先使用 `robopianist.music`，回退到 `pretty_midi`）
- 计算MFCC特征（13维）
- 使用余弦相似度比较两个音频
- 最终评分：`(1 - audio_weight) * DTW_score + audio_weight * audio_similarity`

默认 `audio_weight = 0.2`（即20%权重给音频相似度）

### 评分等级

- **Excellent** (>0.8): 优秀演奏
- **Good** (0.6-0.8): 良好演奏
- **Fair** (0.4-0.6): 一般演奏
- **Poor** (<0.4): 需要改进

## 技术细节

### DTW对齐

使用 `librosa.sequence.dtw` 进行对齐，支持以下步长：
- `(1, 1)`: 同时前进
- `(0, 1)`: 只前进参考序列
- `(1, 0)`: 只前进演奏序列

### 特征表示

每个音符由3个特征表示：
- 开始时间 (start)
- 音高 (pitch)
- 时长 (duration)

### 音频相似度

音频相似度使用MFCC（Mel频率倒谱系数）特征比较视频音频和参考MIDI合成音频：
- **视频音频提取**: 使用 `librosa` 从视频中提取单声道音频
- **MIDI音频合成**: 使用 `robopianist.music` 或 `pretty_midi.fluidsynth` 合成参考音频
- **采样率**: 44100 Hz
- **MFCC维度**: 13维
- **Hop length**: 512 samples
- 对时间维度取平均，得到固定长度的特征向量
- 使用余弦相似度度量两个音频的特征相似性

### 评分流程

1. **输入**: 视频文件（MP4等）+ 演奏MIDI文件 + 参考MIDI文件
2. **提取特征**: 
   - 从视频中提取音频波形
   - 从MIDI文件中提取音符特征和合成音频
3. **对齐**: 使用DTW对齐演奏MIDI和参考MIDI的音符序列
4. **比较**: 计算DTW对齐评分和音频相似度评分
5. **综合**: 加权组合两种评分得到最终分数

### 依赖库

- `pretty_midi`: MIDI文件处理
- `librosa`: 音频处理和DTW对齐
- `numpy`: 数值计算
- `scipy`: 距离计算
- `robopianist.music` (可选): 高质量MIDI合成

## 注意事项

- **必需输入**: 视频文件、演奏MIDI文件、参考MIDI文件
- 脚本只评估第一个乐器的音符
- MIDI文件必须有至少一个乐器和音符
- 完美匹配会得到 1.0 分
- 没有音符的演奏会得到 0.0 分
- 如果音频提取/合成失败，脚本会回退到纯DTW评分
- 视频必须包含有效的音频轨道

