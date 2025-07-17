import os

# 定义输出目录
output_dir = "llmmm_files"
os.makedirs(output_dir, exist_ok=True)

# 基础模板
template = """#include "llmmm/llmmm.cuh"

namespace LLMMM {{

class UnalignedM_{0}_MMInstantiatorWrapper: public MMInstantiatorWrapper {{
  MMInstantiator<{0}> mm_instantiator;

public:
  UnalignedM_{0}_MMInstantiatorWrapper()
  {{
    mm_instantiator.apply();
  }}
}} __UnalignedM_{0}_MMInstantiatorWrapper__;

}}  // namespace LLMMM
"""

# 生成1到127的文件
for i in range(1, 128):
    # 使用三位数字格式，不足补零
    num_str = f"{i:03d}"
    filename = f"llmmm_unaligned_m_{num_str}.cu"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(template.format(i))
    
    print(f"生成文件: {filepath}")    
