#!/usr/bin/env python3
import os
import shutil
import subprocess

def reduce_dataset(data_dir, affinity_file, target_count=500):
    """
    减少数据集到指定数量，确保复合物和亲和力数据对应
    """
    print(f"开始处理数据集，目标保留 {target_count} 个样本...")
    
    # 1. 获取当前所有复合物文件夹
    refined_set_dir = os.path.join(data_dir, "refined-set")
    all_complexes = [d for d in os.listdir(refined_set_dir) if os.path.isdir(os.path.join(refined_set_dir, d)) and not d.startswith('.')]
    
    print(f"当前总共有 {len(all_complexes)} 个复合物")
    
    # 2. 读取亲和力文件，提取可用的复合物ID
    available_complexes = set()
    affinity_data = {}
    
    with open(affinity_file, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                pdb_id = parts[0]
                binding_affinity = parts[3]
                available_complexes.add(pdb_id)
                affinity_data[pdb_id] = line.strip()
    
    print(f"亲和力文件中有 {len(available_complexes)} 个复合物记录")
    
    # 3. 找到既有结构文件又有亲和力数据的复合物
    valid_complexes = []
    for complex_id in all_complexes:
        if complex_id in available_complexes:
            # 检查是否有必要的文件
            complex_dir = os.path.join(refined_set_dir, complex_id)
            required_files = [
                f"{complex_id}_ligand.sdf",
                f"{complex_id}_protein.pdb"
            ]
            if all(os.path.exists(os.path.join(complex_dir, f)) for f in required_files):
                valid_complexes.append(complex_id)
    
    print(f"有效的复合物数量（既有结构文件又有亲和力数据）: {len(valid_complexes)}")
    
    # 4. 选择前target_count个复合物
    selected_complexes = sorted(valid_complexes)[:target_count]
    print(f"选择保留的复合物数量: {len(selected_complexes)}")
    
    # 5. 创建备份
    backup_dir = "data_backup"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print("创建备份目录...")
        shutil.copytree(refined_set_dir, os.path.join(backup_dir, "refined-set"))
        shutil.copy2(affinity_file, os.path.join(backup_dir, "INDEX_refined_data.2020"))
        print("备份完成")
    
    # 6. 删除不需要的复合物文件夹
    removed_count = 0
    for complex_id in all_complexes:
        if complex_id not in selected_complexes:
            complex_path = os.path.join(refined_set_dir, complex_id)
            if os.path.isdir(complex_path):
                shutil.rmtree(complex_path)
                removed_count += 1
    
    print(f"删除了 {removed_count} 个复合物文件夹")
    
    # 7. 更新亲和力文件
    new_affinity_file = affinity_file + ".new"
    with open(new_affinity_file, 'w') as f:
        # 写入头部注释
        f.write("# ==============================================================================\n")
        f.write("# List of the protein-ligand complexes in the PDBbind refined set v.2020\n")
        f.write(f"# {len(selected_complexes)} protein-ligand complexes in total, which are ranked by binding data\n")
        f.write("# Latest update: July 2021\n")
        f.write("# PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name\n")
        f.write("# ==============================================================================\n")
        
        # 写入选中的复合物数据
        for complex_id in selected_complexes:
            if complex_id in affinity_data:
                f.write(affinity_data[complex_id] + "\n")
    
    # 替换原文件
    shutil.move(new_affinity_file, affinity_file)
    print(f"更新亲和力文件，现在包含 {len(selected_complexes)} 个条目")
    
    # 8. 验证结果
    remaining_complexes = [d for d in os.listdir(refined_set_dir) if os.path.isdir(os.path.join(refined_set_dir, d)) and not d.startswith('.')]
    print(f"处理完成！剩余复合物数量: {len(remaining_complexes)}")
    
    return selected_complexes

if __name__ == "__main__":
    data_dir = "data"
    affinity_file = "data/INDEX_refined_data.2020"
    target_count = 100
    
    selected = reduce_dataset(data_dir, affinity_file, target_count)
    print(f"\n保留的复合物列表（前10个）: {selected[:10]}")
    print(f"总计: {len(selected)} 个复合物") 