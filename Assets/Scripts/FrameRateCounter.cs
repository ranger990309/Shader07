using UnityEngine;       // 导入Unity核心库
using TMPro;             // 导入TextMeshPro库，用于高质量文本渲染

/// <summary>
/// 帧率计数器组件，用于监测和显示游戏性能数据
/// </summary>
public class FrameRateCounter : MonoBehaviour {

	// 用于显示帧率信息的UI文本组件
	[SerializeField]
	TextMeshProUGUI display;

	// 定义显示模式枚举：FPS(每秒帧数)或MS(每帧毫秒数)
	public enum DisplayMode { FPS, MS }

	// 当前显示模式，默认为FPS
	[SerializeField]
	DisplayMode displayMode = DisplayMode.FPS;

	// 样本持续时间，决定多久更新一次显示数据
	[SerializeField, Range(0.1f, 2f)]
	float sampleDuration = 1f;

	// 当前样本周期内的帧数
	int frames;

	// 当前样本周期的总持续时间，以及最佳(最短)和最差(最长)帧时间
	float duration, bestDuration = float.MaxValue, worstDuration;

	/// <summary>
	/// 每帧更新，收集和显示帧率数据
	/// </summary>
	void Update () {
		// 获取当前帧的持续时间(不受时间缩放影响)
		float frameDuration = Time.unscaledDeltaTime;
		// 帧数递增
		frames += 1;
		// 累加总持续时间
		duration += frameDuration;

		// 更新最佳帧时间(最短的帧时间 = 最高帧率)
		if (frameDuration < bestDuration) {
			bestDuration = frameDuration;
		}
		// 更新最差帧时间(最长的帧时间 = 最低帧率)
		if (frameDuration > worstDuration) {
			worstDuration = frameDuration;
		}

		// 当收集的样本时间达到或超过设定的样本持续时间时，更新显示
		if (duration >= sampleDuration) {
			if (displayMode == DisplayMode.FPS) {
				// FPS模式: 显示最高、平均和最低FPS
				display.SetText(
					"FPS\n{0:0}\n{1:0}\n{2:0}",
					1f / bestDuration,        // 最高FPS (来自最短帧时间)
					frames / duration,        // 平均FPS (总帧数/总时间)
					1f / worstDuration        // 最低FPS (来自最长帧时间)
				);
			}
			else {
				// MS模式: 显示最低、平均和最高帧时间(毫秒)
				display.SetText(
					"MS\n{0:1}\n{1:1}\n{2:1}",
					1000f * bestDuration,     // 最短帧时间(毫秒)
					1000f * duration / frames, // 平均帧时间(毫秒)
					1000f * worstDuration     // 最长帧时间(毫秒)
				);
			}
			
			// 重置所有计数器，准备下一个样本周期
			frames = 0;
			duration = 0f;
			bestDuration = float.MaxValue;
			worstDuration = 0f;
		}
	}
}

/*
 * ==========================================================================
 * 代码分析:
 * ==========================================================================
 * 
 * FrameRateCounter组件概述:
 * -----------------------
 * 这个组件提供了一个实时性能监控工具，可以在游戏运行时显示帧率相关信息。
 * 它有两种显示模式(FPS和MS)，能够跟踪最佳、平均和最差性能指标。
 *
 * 主要功能:
 * --------
 * 1. 性能监控 - 跟踪每帧的执行时间，计算性能指标
 * 2. 双模式显示 - FPS模式(每秒帧数)和MS模式(每帧毫秒数)
 * 3. 可配置采样 - 通过sampleDuration参数调整数据更新频率
 * 4. 实时数据 - 使用unscaledDeltaTime确保准确性
 *
 * 性能指标说明:
 * -----------
 * - 最高FPS/最低MS: 表示最佳性能时刻
 * - 平均FPS/MS: 整体性能状况
 * - 最低FPS/最高MS: 表示性能瓶颈或卡顿
 *
 * 使用场景:
 * --------
 * - 开发过程中监控性能变化
 * - 识别潜在的性能瓶颈
 * - 测试优化效果
 * - 在不同设备上评估性能表现
 */