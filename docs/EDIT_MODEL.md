# Hakyking 非破坏性编辑模型

编辑状态以 `AudioSliceEditModel` 为唯一聚合入口：

1. `SyllableClipModel` 保存源音频区域、时间线位置、目标时长和音高中心。
2. `PitchAutomationModel` 只描述音高自动化，不拥有音频，也不负责切片。
3. `PitchControlPoint` 是曲线编辑标记；添加、移动或删除控制点不会切断源音频。
4. `VibratoRegion` 是叠加在曲线区间上的参数化效果，保存区间、周期、深度、相位和波形。
5. `SliceRenderRequest` 是 UI、工程文件和 DSP 之间的统一只读渲染快照。

`AudioSliceGraphicsItem` 仍保留旧属性名作为兼容代理，但真实值由上述模型持有。这样可以逐步把交互代码从巨大的视图类迁出，而不破坏现有工程和快捷键。

## V / B / N 到模型的映射

- V 只改变 `PitchAutomationModel.control_points` 中既有点的位置或 UI 选择状态。
- B 只创建/删除 `PitchControlPoint`，不会改变音频切片边界。
- 两个及以上由 V 选中的点会派生临时曲线选择区；选择区会按原始 F0 中的无声断点拆开，不写入工程文件。
- N 只在这些临时选择区创建或更新 `VibratoRegion`。颤音保持参数化，不展开成大量控制点。
- 所有持久变化都通过 `edit_state()` 前后快照进入 `ChangeParameterCommand`，UI 选择本身不进入工程文件或撤销栈。
