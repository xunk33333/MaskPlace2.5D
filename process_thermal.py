"""热分析处理脚本,用于生成20个布局的热点分析结果"""

import os
import numpy as np
from typing import List, Tuple
from Thermal import Thermal_solver
import time

def parse_intpsize_file(intpsize_path: str) -> Tuple[float, float]:
    """解析.intpsize文件获取尺寸参数(单位:um)"""
    try:
        with open(intpsize_path) as f:
            line = f.readline().strip()
            width = height = float(line)
            return width, height
    except Exception as e:
        print(f"Error reading {intpsize_path}: {str(e)}")
        raise

def get_system_params(case_dir: str) -> object:
    """获取系统参数配置"""
    class SystemParams:
        def __init__(self):
            self.granularity = 1000  # 固定值1000um
            # 从.intpsize文件读取尺寸
            case_name = os.path.basename(case_dir)
            intpsize_path = os.path.join(case_dir, f"{case_name}.intpsize")
            self.intp_width, self.intp_height = parse_intpsize_file(intpsize_path)
    
    return SystemParams()

def parse_pl_file(pl_path: str) -> Tuple[List[Tuple[float, float]], List[float], List[float]]:
    """解析.pl布局文件"""
    chip_names = []
    positions = []
    widths = []
    heights = []
    
    try:
        with open(pl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                x, y = float(parts[1]), float(parts[2])  # 左下角坐标(um)
                w, h = float(parts[3]), float(parts[4])  # 宽高(um)
                
                # 转换为中心坐标(mm)
                center_x = (x + w/2)
                center_y = (y + h/2)
                
                chip_names.append(parts[0])
                positions.append((center_x, center_y))
                widths.append(w)
                heights.append(h)
    
    except Exception as e:
        print(f"Error parsing {pl_path}: {str(e)}")
        raise
    
    return chip_names, positions, widths, heights

def parse_power_file(power_path: str) -> List[float]:
    """解析.power文件"""
    powers = {}
    with open(power_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                powers[parts[0]] = float(parts[1])
    return powers

def main(case_name = "Case1"):
    """主处理流程"""
    # 配置参数
    case_dir = f"./cases/{case_name}"
    output_dir = f"./dataset/{case_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化系统参数(从.intpsize读取)
    system = get_system_params(case_dir)
    
    # 初始化热分析器
    thermal = Thermal_solver(os.path.join(output_dir, ""))
    thermal.set_params(system)
    
    # 处理200个布局
    for idx in range(1, 201):
        try:
            start = time.time()
            # 文件路径
            pl_file = os.path.join(case_dir, f"{case_name}_{idx}.pl")
            power_file = os.path.join(case_dir, f"{case_name}.power")
            
            # 解析文件
            chip_names, positions, widths, heights = parse_pl_file(pl_file)
            powers = parse_power_file(power_file)
            powers = [powers.get(name, 0.0) for name in chip_names] 
            # 设置热分析参数
            thermal.set_pos(
                np.array(powers),
                np.array(positions).transpose(),
                np.array(widths),
                np.array(heights))
            
            # 执行热分析
            output_prefix = f"gen_dataset_{idx}"
            thermal.run_hotspot(output_prefix)
            print(f"已完成布局 {case_name}:{idx}/20")
            end = time.time()
            print(f"Time taken for layout {idx}: {end - start:.6f} seconds")
        except Exception as e:
            print(f"处理布局{idx}时出错: {str(e)}")
            continue

if __name__ == "__main__":
    for idx in range(1,11):
        main(case_name = f"Case{idx}")
