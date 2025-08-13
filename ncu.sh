METRICS=(
  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed
  smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed
  smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed
  smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed
  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
  smsp__sass_thread_inst_executed_op_fmul_pred_on.sum
  smsp__sass_thread_inst_executed_op_dadd_pred_on.sum
  smsp__sass_thread_inst_executed_op_dmul_pred_on.sum
  sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum.per_cycle_elapsed
  smsp__cycles_elapsed.avg.per_second
  derived__l1tex__lsu_writeback_bytes_mem_lg.sum.per_second
  derived__lts__lts2xbar_bytes.sum.per_second
  dram__bytes.sum.per_second
)

SECTIONS=(
    --section ComputeWorkloadAnalysis
    --section InstructionStats           
    --section SchedulerStats           
    --section WarpStateStats
    --section LaunchStats                
    --section MemoryWorkloadAnalysis_Tables
    --section SourceCounters
    --section SpeedOfLight
    --section SpeedOfLight_HierarchicalDoubleRooflineChart
    --section SpeedOfLight_HierarchicalHalfRooflineChart
    --section SpeedOfLight_HierarchicalSingleRooflineChart
    --section SpeedOfLight_HierarchicalTensorRooflineChart
    --section SpeedOfLight_RooflineChart
    --section WorkloadDistribution
    --section PmSampling
)

# 用逗号连接指标列表
METRICS_STR=$(IFS=,; echo "${METRICS[*]}")

# 执行命令
/usr/local/NVIDIA-Nsight-Compute-2025.2/ncu \
    -k 'regex:^(?!.*naive).*$' \
    --metrics "$METRICS_STR" \
    "${SECTIONS[@]}" \
    -f -o "${1}.ncu-rep" "${1}"
