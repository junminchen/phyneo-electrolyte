#!/usr/bin/env python
"""
EAPNN (Equivariant Atom-Pair Neural Network) 训练脚本
用于训练基于原子对特征的神经网络模型来预测分子间相互作用能
"""

# =============================================================================
# 标准库导入
# =============================================================================
import os
import sys
import time
import json
import glob
import pickle
import argparse
from typing import Tuple, Dict, List, Any
from functools import partial

# =============================================================================
# 科学计算库导入
# =============================================================================
import numpy as np
import pandas as pd
import scipy
np.seterr(divide='ignore', invalid='ignore')

# =============================================================================
# JAX相关导入
# =============================================================================
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
from flax import linen as nn
from flax.training import train_state
import optax
from jax import config
config.update("jax_debug_nans", True)  # Enable NaN checking

# =============================================================================
# 分子动力学和化学信息学库
# =============================================================================
from dmff.api import Hamiltonian
from dmff.utils import jit_condition, regularize_pairs, pair_buffer_scales
from dmff.admp.pairwise import distribute_scalar, distribute_v3
from dmff.admp.spatial import pbc_shift
from dmff.common import nblist

# OpenMM相关
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *

# 分子分析库
import MDAnalysis as mda
import mdtraj as md
from ase.io import read, write

# =============================================================================
# 机器学习库
# =============================================================================
import torch
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 可视化库
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from IPython.display import clear_output

# =============================================================================
# 全局配置
# =============================================================================
# JAX配置（可选）
# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", False)

# 原子类型定义
ZINDEX = [1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 15.0, 16.0]

# 电荷到索引映射
CHARGE_TO_INDEX = {
    0.0: 100000, 1.0: 0, 3.0: 1, 5.0: 2, 6.0: 3, 7.0: 4,
    8.0: 5, 9.0: 6, 11.0: 7, 15.0: 8, 16.0: 9,
}


# =============================================================================
# 工具函数
# =============================================================================


def print_training_progress(
    epoch, num_epochs, loss, loss_history_train, loss_history_test,
    true_energies, predicted_energies, base_energies, distances,
    epoch_time, total_time, figure_size=(13, 4), english_font=None):
    """
    打印训练进度信息
    
    Args:
        epoch: 当前epoch
        num_epochs: 总epoch数
        loss: 当前损失
        loss_history_train: 训练损失历史
        loss_history_test: 测试损失历史
        true_energies: 真实能量
        predicted_energies: 预测能量
        base_energies: 基准能量
        distances: 距离
        epoch_time: epoch时间
        total_time: 总时间
        figure_size: 图形尺寸
        english_font: 英文字体
    """
    if english_font is None:
        english_font = fm.FontProperties(family='DejaVu Sans', size=12)
    
    # 转换系数
    KJ_TO_KCAL = 0.239006

    # 转换单位
    true_energies = true_energies * KJ_TO_KCAL
    predicted_energies = predicted_energies * KJ_TO_KCAL
    base_energies = base_energies * KJ_TO_KCAL
    loss_history_train = [l * KJ_TO_KCAL for l in loss_history_train]
    loss_history_test = [l * KJ_TO_KCAL for l in loss_history_test]
    loss = loss * KJ_TO_KCAL

    # 计算误差指标
    rmse = np.sqrt(np.mean((true_energies - predicted_energies)**2))
    rmse_base = np.sqrt(np.mean((true_energies - base_energies)**2))
    prediction_errors = predicted_energies - true_energies
    base_errors = base_energies - true_energies
    mean_error = np.mean(prediction_errors)
    std_error = np.std(prediction_errors)

    # 计算平均epoch时间
    avg_epoch_time = total_time / (epoch + 1)

    print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}, RMSE: {rmse:.4f}, " +
        f"Epoch time: {epoch_time:.2f}s, Avg epoch time: {avg_epoch_time:.2f}s")


def plot_training_progress(
    epoch, num_epochs, loss, loss_history_train, loss_history_test,
    true_energies, predicted_energies, base_energies, distances,
    epoch_time, total_time, figure_size=(13, 4), english_font=None):
    """
    绘制训练进度图表
    
    包含三个子图：损失曲线、预测对比、误差分布
    
    Args:
        epoch: 当前epoch
        num_epochs: 总epoch数
        loss: 当前损失
        loss_history_train: 训练损失历史
        loss_history_test: 测试损失历史
        true_energies: 真实能量
        predicted_energies: 预测能量
        base_energies: 基准能量
        distances: 距离
        epoch_time: epoch时间
        total_time: 总时间
        figure_size: 图形尺寸
        english_font: 英文字体
    
    Returns:
        tuple: (图形对象, 轴对象列表)
    """
    if english_font is None:
        english_font = fm.FontProperties(family='DejaVu Sans', size=12)
    
    # 转换系数
    KJ_TO_KCAL = 0.239006

    # 转换单位
    true_energies = true_energies * KJ_TO_KCAL
    predicted_energies = predicted_energies * KJ_TO_KCAL
    base_energies = base_energies * KJ_TO_KCAL
    loss_history_train = [l * KJ_TO_KCAL for l in loss_history_train]
    loss_history_test = [l * KJ_TO_KCAL for l in loss_history_test]
    loss = loss * KJ_TO_KCAL

    # 计算误差指标
    rmse = np.sqrt(np.mean((true_energies - predicted_energies)**2))
    rmse_base = np.sqrt(np.mean((true_energies - base_energies)**2))
    prediction_errors = predicted_energies - true_energies
    base_errors = base_energies - true_energies
    mean_error = np.mean(prediction_errors)
    std_error = np.std(prediction_errors)

    # 计算平均epoch时间
    avg_epoch_time = total_time / (epoch + 1)

    # 创建图表
    fig, axs = plt.subplots(1, 3, figsize=figure_size)

    # 1. 损失下降曲线
    ax1 = axs[0]
    ax1.plot(range(1, len(loss_history_train) + 1), loss_history_train, marker='s', color='tomato',
            linewidth=2, markersize=5, markeredgecolor='k', markeredgewidth=0.8, label='Training')
    ax1.set_xlabel('Epochs', fontproperties=english_font, labelpad=8)
    ax1.set_ylabel('Training Loss (kcal/mol)', color='tomato', fontproperties=english_font, labelpad=8)
    ax1.set_title(f"Training Avg Epoch time: {avg_epoch_time:.1f}s | Total: {total_time/60:.1f}min",
                fontproperties=english_font, pad=10)
    ax1.tick_params(axis='y', labelcolor='tomato')
    ax1.tick_params(axis='both', which='major',
                length=6, direction='out', width=1.2,
                bottom=True, top=False, left=True, right=False)
    ax1.tick_params(axis='both', which='minor',
                length=3, direction='out', width=1.0,
                bottom=True, top=False, left=True, right=False)

    # 创建次要y轴
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(loss_history_test) + 1), loss_history_test, marker='o', color='cornflowerblue',
            linewidth=2, markersize=5, markeredgecolor='k', markeredgewidth=0.8, label='Testing')
    ax2.set_ylabel('Testing Loss (kcal/mol)', color='cornflowerblue', fontproperties=english_font, labelpad=8)
    ax2.tick_params(axis='y', labelcolor='cornflowerblue')
    ax2.tick_params(axis='both', which='major',
                length=6, direction='out', width=1.2,
                bottom=True, top=False, left=False, right=True)
    ax2.tick_params(axis='both', which='minor',
                length=3, direction='out', width=1.0,
                bottom=True, top=False, left=False, right=True)

    # 添加次要刻度
    for ax in [ax1, ax2]:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # 添加网格和图例
    ax1.grid(True, linestyle='--', alpha=0.7)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', prop=english_font)

    if loss_history_test:
        ax1.text(0.05, 0.95, f'Current loss: {loss_history_test[-1]:.4f}',
                transform=ax1.transAxes, fontproperties=english_font, verticalalignment='top')

    # 2. 预测值与真实值的对角线图
    ax = axs[1]
    sc2 = ax.scatter(true_energies, base_energies, alpha=0.4,
                    label='w/o corr', color='lightgray', edgecolor='k', linewidth=0.5)
    sc1 = ax.scatter(true_energies, predicted_energies, alpha=0.7,
                    label='w/ corr', color='tab:blue', edgecolor='k', linewidth=0.5)

    min_val = min(np.min(true_energies), np.min(predicted_energies))
    max_val = max(np.max(true_energies), np.max(predicted_energies))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    ax.set_xlabel("True Energy (kcal/mol)", fontproperties=english_font, labelpad=8)
    ax.set_ylabel("Predicted Energy (kcal/mol)", fontproperties=english_font, labelpad=8)
    ax.set_title("Parity Plot: Predicted vs True Energy", fontproperties=english_font, pad=10)

    ax.tick_params(axis='both', which='major',
                length=6, direction='out', width=1.2,
                bottom=True, top=False, left=True, right=False)
    ax.tick_params(axis='both', which='minor',
                length=3, direction='out', width=1.0,
                bottom=True, top=False, left=True, right=False)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.text(0.05, 0.95, f'RMSE w/ corr: {rmse:.4f}',
        transform=ax.transAxes, fontproperties=english_font, verticalalignment='top')
    ax.text(0.05, 0.90, f'RMSE w/o corr: {rmse_base:.4f}',
        transform=ax.transAxes, fontproperties=english_font, verticalalignment='top')
    ax.legend(loc='lower right', prop=english_font)

    # 3. 预测误差与距离的关系图
    ax = axs[2]
    sc_base = ax.scatter(distances, base_errors, alpha=0.4,
                        color='red', edgecolor='darkred', linewidth=0.5,
                        marker='^', label='w/o corr')
    sc = ax.scatter(distances, prediction_errors, alpha=0.7,
                c=np.abs(prediction_errors), cmap='Blues',
                edgecolor='k', linewidth=0.5, marker='o', label='w/ corr')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)

    ax.set_xlabel("Distance", fontproperties=english_font, labelpad=8)
    ax.set_ylabel("Prediction Error (kcal/mol)", fontproperties=english_font, labelpad=8)
    ax.set_title("Error Distribution vs Distance", fontproperties=english_font, pad=10)

    ax.tick_params(axis='both', which='major',
                length=6, direction='out', width=1.2,
                bottom=True, top=False, left=True, right=False)
    ax.tick_params(axis='both', which='minor',
                length=3, direction='out', width=1.0,
                bottom=True, top=False, left=True, right=False)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.legend(loc='lower right', prop=english_font)
    ax.text(0.05, 0.95, f'Mean error: {mean_error:.4f}',
        transform=ax.transAxes, fontproperties=english_font, verticalalignment='top')
    ax.text(0.05, 0.90, f'Std error: {std_error:.4f}',
        transform=ax.transAxes, fontproperties=english_font, verticalalignment='top')

    # 添加颜色条
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('|Error| Magnitude (kcal/mol)', fontproperties=english_font)
    cbar.ax.tick_params(length=6, width=1.2)
    cbar.ax.tick_params(axis='y', which='major',
                    length=6, direction='in', width=1.2,
                    left=False, right=True)
    cbar.ax.tick_params(axis='y', which='minor',
                    length=3, direction='in', width=1.0,
                    left=False, right=True)
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    cbar.ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # 设置边框样式
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.spines[['right', 'top']].set_visible(True)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout(pad=1.2)

    return fig, axs

def get_data(data, arr):
    """
    从数据中筛选指定数组的测试数据
    
    Args:
        data: 包含键值对的数据字典
        arr: 筛选条件数组
    
    Returns:
        list: 筛选后的键列表
    """
    dimer_test = [key for key in data if key.split('_')[-2] in arr and key.split('_')[-1] in arr]
    return dimer_test


# =============================================================================
# 数据集类
# =============================================================================
class MoleculeTorchDataset(Dataset):
    """
    分子数据集类，适配无效索引处理逻辑，确保数据与模型兼容
    仅填充原子数（行），固定邻居数（列）为max_neighbors
    """
    
    def __init__(self, ase_structures, max_atoms=60, max_neighbors=10):
        """
        初始化数据集
        
        Args:
            ase_structures: ASE结构列表
            max_atoms: 最大原子数（固定行）
            max_neighbors: 最大邻居数（固定列）
        """
        self.max_atoms = max_atoms
        self.max_neighbors = max_neighbors  # 全局固定邻居列数
        self.structures = ase_structures
        
        # 预检查：验证原子类型和邻居列表列数
        self._validate_atom_types()
        self._validate_neighbor_columns()
    
    def _validate_atom_types(self):
        """验证所有结构的原子类型索引非负"""
        for idx, structure in enumerate(self.structures):
            atypes = structure.get_array('atype')
            if len(atypes) == 0:
                raise ValueError(f"结构 {idx} 的原子类型数组为空")
            if np.any(atypes < 0):
                raise ValueError(f"结构 {idx} 包含无效原子类型索引（负数）: {atypes[atypes < 0]}")
    
    def _validate_neighbor_columns(self):
        """验证所有结构的topo_nblist列数不超过max_neighbors"""
        for idx, structure in enumerate(self.structures):
            nblist = np.array(structure.info['topo_nblist'])
            if nblist.size == 0:
                continue  # 空列表后续会自动初始化
            current_cols = nblist.shape[1]
            if current_cols > self.max_neighbors:
                raise ValueError(
                    f"结构 {idx} 的邻居列数 {current_cols} 超过max_neighbors {self.max_neighbors}，"
                    f"请增大max_neighbors或截断原始数据"
                )
    
    def __len__(self):
        return len(self.structures)
    
    def __getitem__(self, idx):
        structure = self.structures[idx]
        n_atoms = len(structure)
        
        # 1. 位置信息（填充至max_atoms行）
        pos = np.pad(
            structure.get_positions(), 
            ((0, self.max_atoms - n_atoms), (0, 0)), 
            mode='constant', constant_values=0.0
        )
        
        # 2. 盒子信息
        box = np.array(structure.get_cell(), dtype=np.float64)
        
        # 3. 原子序数（填充至max_atoms）
        atomic_nums = np.pad(
            structure.get_atomic_numbers(), 
            (0, self.max_atoms - n_atoms), 
            mode='constant', constant_values=0
        )
        
        # 4. 能量信息
        energy = float(structure.get_potential_energy())
        sr_energy = float(structure.info['sr_energy'])
        distance = float(structure.info['distance'])
        
        # 5. 原子掩码（1=有效原子）
        mask = np.pad(
            np.ones(n_atoms, dtype=np.int32), 
            (0, self.max_atoms - n_atoms), 
            mode='constant', constant_values=0
        )
        
        # 6. 分子ID（填充无效值10000）
        mol_ID = np.pad(
            structure.get_array('molID'), 
            (0, self.max_atoms - n_atoms), 
            mode='constant', constant_values=10000
        )
        
        # 7. 原子对列表
        pairs = np.array(structure.info['pairs'], dtype=np.int32)
        
        # 8. 拓扑邻居列表和掩码（核心处理）
        orig_topo_nblist = np.array(structure.info['topo_nblist'], dtype=np.int32)
        orig_topo_mask = np.array(structure.info['topo_mask'], dtype=np.int32)
        
        # 处理空列表：初始化(n_atoms, max_neighbors)的数组
        if orig_topo_nblist.size == 0:
            orig_topo_nblist = np.full((n_atoms, self.max_neighbors), -1, dtype=np.int32)
            orig_topo_mask = np.zeros((n_atoms, self.max_neighbors), dtype=np.int32)
        else:
            # 填充列至max_neighbors（若原始列数不足）
            current_cols = orig_topo_nblist.shape[1]
            if current_cols < self.max_neighbors:
                orig_topo_nblist = np.pad(
                    orig_topo_nblist,
                    ((0, 0), (0, self.max_neighbors - current_cols)),  # 仅填充列
                    mode='constant', constant_values=-1
                )
                orig_topo_mask = np.pad(
                    orig_topo_mask,
                    ((0, 0), (0, self.max_neighbors - current_cols)),  # 仅填充列
                    mode='constant', constant_values=0
                )
        
        # 填充行至max_atoms（仅扩展原子数维度）
        topo_nblist = np.pad(
            orig_topo_nblist,
            ((0, self.max_atoms - n_atoms), (0, 0)),  # 仅填充行
            mode='constant', constant_values=-1
        )
        topo_mask = np.pad(
            orig_topo_mask,
            ((0, self.max_atoms - n_atoms), (0, 0)),  # 仅填充行
            mode='constant', constant_values=0
        )
        
        # 验证有效索引不越界
        valid_mask = topo_mask == 1
        valid_indices = topo_nblist[valid_mask]
        if len(valid_indices) > 0 and np.any(valid_indices >= n_atoms):
            invalid = valid_indices[valid_indices >= n_atoms]
            raise ValueError(f"结构 {idx} 的topo_nblist包含无效索引 {invalid}，超过实际原子数 {n_atoms}")
        
        # 9. 原子类型索引（填充至max_atoms）
        atypes = np.pad(
            structure.get_array('atype'),
            (0, self.max_atoms - n_atoms),
            mode='constant', constant_values=0  # 填充原子用默认类型0
        )
        # atypes = np.pad(
        #     structure.get_array('atype') - 1,  # 关键：1基转0基
        #     (0, self.max_atoms - n_atoms),
        #     mode='constant', constant_values=0  # 填充原子用默认类型0（0基）
        # )        
        return {
            'pos': pos.astype(np.float64),
            'box': box.astype(np.float64),
            'atomic_numbers': atomic_nums.astype(np.int32),
            'energy': energy,
            'sr_energy': sr_energy,
            'mask': mask.astype(np.int32),
            'molID': mol_ID.astype(np.int32),
            'pairs': pairs.astype(np.int32),
            'atypes': atypes.astype(np.int32),
            'distance': distance,
            'topo_mask': topo_mask.astype(np.int32),
            'topo_nblist': topo_nblist.astype(np.int32),
        }
# =============================================================================
# 训练相关函数
# =============================================================================

def torch_batch_to_jax(batch):
    """
    将PyTorch批次转换为JAX数组
    
    Args:
        batch: PyTorch批次字典
    
    Returns:
        dict: JAX批次字典
    """
    jax_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            jax_batch[k] = jnp.array(v.numpy())
        else:
            jax_batch[k] = jnp.array(v)
    return jax_batch


def create_train_state(model, learning_rate, key):
    """
    创建训练状态
    
    Args:
        model: 神经网络模型
        learning_rate: 学习率
        key: JAX随机密钥
    
    Returns:
        TrainState: 训练状态对象
    """
    params = model.init(key, 
                    jnp.array(batch['pos'][0]), 
                    jnp.array(batch['box'][0]), 
                    jnp.array(batch['pairs'][0]), 
                    jnp.array(batch['topo_nblist'][0]),
                    jnp.array(batch['topo_mask'][0]),
                    jnp.array(batch['molID'][0]),
                    jnp.array(batch['atypes'][0]),)
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    """
    单步训练函数
    
    Args:
        state: 训练状态
        batch: 训练批次
    
    Returns:
        tuple: (更新后的状态, 损失值)
    """
    def loss_fn(params):
        pred_delta = jax.vmap(state.apply_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
            params, 
            batch['pos'], 
            batch['box'], 
            batch['pairs'], 
            batch['topo_nblist'],
            batch['topo_mask'],
            batch['molID'],
            batch['atypes'],
        )            
        true_delta = batch['energy']  # 真实的力场-DFT差值
        
        # Huber损失参数（delta=10，可根据你的差值标准差调整）
        error = pred_delta - true_delta
        huber_loss = jnp.where(
            jnp.abs(error) <= 10,
            0.5 * jnp.square(error),  # 小误差：MSE
            10 * (jnp.abs(error) - 0.5 * 10)  # 大误差：MAE
        )
        return jnp.mean(huber_loss)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def get_topology_neighbors(pdb_file, connectivity=2, max_neighbors=18, max_n_atoms=None):
    """
    获取分子的拓扑邻居信息，确保输出索引均为有效范围（非负且不超过原子总数）
    
    Args:
        pdb_file: PDB文件路径
        connectivity: 连接性级别（默认2）
        max_neighbors: 最大邻居数（默认18）
        max_n_atoms: 最大原子数（默认None，自动从PDB获取）
    
    Returns:
        tuple: 
            topo_nblist_data: 拓扑邻居列表（无效位置用0填充，配合mask使用）
            neighbor_mask: 邻居掩码（1表示有效邻居，0表示无效）
    """
    mol = mda.Universe(pdb_file)
    n_atoms = len(mol.atoms)  # 实际原子总数（有效索引范围：0 ~ n_atoms-1）
    
    # 确定最大原子数（不超过实际原子数，避免无效扩展）
    if max_n_atoms is None:
        max_n_atoms = n_atoms
    else:
        max_n_atoms = min(max_n_atoms, n_atoms)  # 防止用户输入超过实际原子数的值
    
    # 确定最大邻居数（不超过实际可能的邻居数）
    if max_neighbors is None:
        max_neighbors = np.max([len(fragments) for fragments in mol.atoms.fragments])
    max_neighbors = min(max_neighbors, n_atoms - 1)  # 每个原子最多有n_atoms-1个邻居（排除自身）
    
    # 初始化邻居列表和掩码（用0填充无效位置，后续通过mask区分）
    indices = np.zeros((max_n_atoms, max_neighbors), dtype=np.int32)  # 无效值默认0（合法索引）
    mask = np.zeros((max_n_atoms, max_neighbors), dtype=np.int32)      # 0表示无效
    
    try:
        has_bonds = len(mol.bonds) > 0
    except AttributeError:
        has_bonds = False
    
    if has_bonds:
        # 构建邻接矩阵（记录原子间连接关系）
        adj_matrix = np.zeros((n_atoms, n_atoms), dtype=bool)
        for bond in mol.bonds:
            i, j = bond.atoms[0].index, bond.atoms[1].index
            adj_matrix[i, j] = adj_matrix[j, i] = True
        
        # 计算拓扑距离（通过邻接矩阵幂次扩展连接性）
        adj_matrix_initial = np.copy(adj_matrix)
        adj_matrix_odd = np.copy(adj_matrix)
        adj_matrix_self_even = np.copy(adj_matrix)
        
        for _ in range(connectivity - 1):
            adj_matrix_self_even = np.dot(adj_matrix_self_even, adj_matrix_self_even)
            adj_matrix = adj_matrix_odd | adj_matrix_self_even
            adj_matrix_odd = np.dot(adj_matrix_self_even, adj_matrix_initial)        
        
        # 填充邻居列表和掩码
        for i in range(max_n_atoms):  # 仅遍历有效原子范围
            # 获取所有有效邻居（排除自身）
            neighbors = np.where(adj_matrix[i])[0]
            neighbors = neighbors[neighbors != i]  # 过滤自身索引
            
            # 限制邻居数量不超过max_neighbors
            n_real_neighbors = min(len(neighbors), max_neighbors)
            if n_real_neighbors > 0:
                indices[i, :n_real_neighbors] = neighbors[:n_real_neighbors]  # 有效邻居索引
                mask[i, :n_real_neighbors] = 1  # 标记有效
        
        # 验证所有索引均在有效范围内（防止越界）
        valid_indices = indices[mask == 1]
        if len(valid_indices) > 0:
            assert np.all((valid_indices >= 0) & (valid_indices < n_atoms)), \
                f"拓扑邻居列表包含无效索引！有效范围应为[0, {n_atoms-1}]，但出现{valid_indices[valid_indices >= n_atoms]}"
    
    return indices, mask


@jit_condition(static_argnums=())
@partial(jax.vmap, in_axes=(0, None, None), out_axes=(0, 0, 0, 0))
def get_environment_atoms(pairs, topo_nblist, topo_mask):
    """
    获取原子对的环境原子信息，确保-1无效索引被彻底屏蔽
    
    Args:
        pairs: 原子对索引
        topo_nblist: 拓扑邻居列表（可能包含-1无效索引）
        topo_mask: 拓扑掩码（1表示有效邻居，0表示无效）
    
    Returns:
        tuple: (j_neighbors, k_neighbors, valid_mask_j, valid_mask_k)
    """
    j_centers = pairs[0]  # 中心原子j的索引
    k_centers = pairs[1]  # 中心原子k的索引
    
    # 1. 安全索引：使用jnp.take获取邻居列表
    j_neighbors = jnp.take(topo_nblist, j_centers, axis=0)  # [max_neighbors,]
    k_neighbors = jnp.take(topo_nblist, k_centers, axis=0)  # [max_neighbors,]
    
    # 2. 过滤无效索引（-1表示无效）
    valid_j = j_neighbors != -1  # [max_neighbors,]
    valid_k = k_neighbors != -1  # [max_neighbors,]
    
    # 3. 排除邻居等于中心原子自身的情况
    mask_j = (j_neighbors != j_centers) & (j_neighbors != k_centers) & valid_j
    mask_k = (k_neighbors != j_centers) & (k_neighbors != k_centers) & valid_k
    
    # 4. 结合拓扑掩码（确保原始掩码中的无效值被过滤）
    topo_mask_j = jnp.take(topo_mask, j_centers, axis=0)  # [max_neighbors,]
    topo_mask_k = jnp.take(topo_mask, k_centers, axis=0)  # [max_neighbors,]
    valid_mask_j = topo_mask_j & mask_j  # 最终有效掩码（1=有效）
    valid_mask_k = topo_mask_k & mask_k  # 最终有效掩码（1=有效）
    
    # 关键修改：将无效索引替换为0（合法索引），避免-1进入嵌入层
    # （0是任意合法原子索引，后续会被valid_mask_j/k屏蔽，不影响结果）
    j_neighbors = jnp.where(valid_mask_j, j_neighbors, 0)
    k_neighbors = jnp.where(valid_mask_k, k_neighbors, 0)
    
    return j_neighbors, k_neighbors, valid_mask_j, valid_mask_k


def parameter_shapes(params):
    """获取参数形状的辅助函数"""
    return jax.tree_util.tree_map(lambda p: p.shape, params)


@jit_condition(static_argnums=())
@partial(jax.vmap, in_axes=(0, None), out_axes=0)
def cutoff_cosine(distances, cutoff):
    """
    余弦截止函数（二阶可微）
    
    Args:
        distances: 距离数组
        cutoff: 截止距离
    
    Returns:
        jnp.array: 截止函数值
    """
    x = distances / cutoff
    return jnp.where(x < 1, 0.5 * (jnp.cos(jnp.pi * x) + 1), 0.0)



# =============================================================================
# 神经网络模型类
# =============================================================================

class EAPNNForce(nn.Module):
    """
    EAPNN (Equivariant Atom-Pair Neural Network) 主模型类
    
    结合特征提取器和神经网络来预测分子间相互作用能
    """
    
    n_atype: int          # 原子类型数量
    rc: float            # 截断距离
    n_atoms: int         # 原子数量
    acsf_nmu: int        # 原子中心对称函数参数数量
    apsf_nmu: int        # 原子对对称函数参数数量
    acsf_eta: float      # 原子中心对称函数eta参数
    apsf_eta: float      # 原子对对称函数eta参数
    embed_dim: int = 16  # 嵌入维度（远小于157）
    use_pbc: bool = True # 是否使用周期性边界条件

    def setup(self):
        """初始化模型组件"""
        self.feature_extractor = FeatureExtractor(
            n_atoms=self.n_atoms,
            n_atype=self.n_atype, 
            rc=self.rc, 
            acsf_nmu=self.acsf_nmu,
            apsf_nmu=self.apsf_nmu,
            acsf_eta=self.acsf_eta,
            apsf_eta=self.apsf_eta,
            embed_dim=self.embed_dim,
            use_pbc=self.use_pbc
        )
        self.neural_network = NeuralNetwork()

    def __call__(self, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices):
        """
        前向传播
        
        Args:
            pos: 原子位置 [n_atoms, 3]
            box: 模拟盒子 [3, 3]
            pairs: 原子对索引 [n_pairs, 2]
            topo_nblist: 拓扑邻居列表 [n_atoms, max_neighbors]
            topo_mask: 拓扑掩码 [n_atoms, max_neighbors]
            mol_ID: 分子ID [n_atoms]
            atype_indices: 原子类型索引 [n_atoms]
        
        Returns:
            jnp.array: 预测的总能量
        """
        features, dr_norm, buffer_scales = self.feature_extractor(
            pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices)
        atomic_energies = self.neural_network(features, dr_norm, buffer_scales)
        return jnp.sum(atomic_energies)

    def get_features(self, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices):
        """获取特征"""
        return self.feature_extractor(pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices)

    def get_energy(self, features, dr_norm, buffer_scales):
        """从特征计算能量"""
        atomic_energies = self.neural_network(features, dr_norm, buffer_scales)
        return jnp.sum(atomic_energies)


class FeatureExtractor(nn.Module):
    """
    特征提取器类
    
    负责从原子位置和拓扑信息中提取原子中心对称函数(ACSF)和原子对对称函数(APSF)特征
    """
    
    n_atoms: int         # 原子数量
    n_atype: int         # 原子类型数量
    rc: float           # 截断距离
    acsf_nmu: int = 20  # 原子中心对称函数参数数量
    apsf_nmu: int = 10  # 原子对对称函数参数数量
    acsf_eta: float = 100  # 原子中心对称函数eta参数
    apsf_eta: float = 25   # 原子对对称函数eta参数
    embed_dim: int = 16  # 嵌入维度（远小于157）
    use_pbc: bool = True   # 是否使用周期性边界条件

    def setup(self):
        """初始化特征提取器参数"""
        self.atom_embed = nn.Embed(num_embeddings=self.n_atype, features=self.embed_dim)

        # ACSF参数初始化
        self.acsf_mus = jnp.linspace(0.0, 5.0, self.acsf_nmu)
        
        # APSF参数初始化
        self.apsf_mus = jnp.linspace(-1.0, 1.0, self.apsf_nmu)

    def compute_atomcenter_features(self, pos, box, topo_nblist, topo_mask, atype_indices, acsf_mus, acsf_eta):
        """
        计算原子中心对称函数特征
        
        Args:
            pos: 原子位置 [n_atoms, 3]
            box: 模拟盒子 [3, 3]
            topo_nblist: 拓扑邻居列表 [n_atoms, max_neighbors]
            topo_mask: 拓扑掩码 [n_atoms, max_neighbors]
            atype_indices: 原子类型索引 [n_atoms]
            acsf_mus: ACSF mu参数
            acsf_eta: ACSF eta参数
        
        Returns:
            jnp.array: 原子中心特征 [n_atoms, n_mu, n_atype]
        """
        # 获取环境原子位置
        r_center = pos  # [n_atoms, 3]
        r_env = pos[topo_nblist]  # [n_atoms, max_neighbors, 3]
        
        # 计算相对位置和距离
        dr = r_env - r_center[:, None, :]  # [n_atoms, max_neighbors, 3]
        box_inv = jnp.linalg.inv(box)
        dr = pbc_shift(dr, box, box_inv)  # 周期性边界条件处理
        dr_norm = jnp.linalg.norm(dr+1e-10, axis=2)  # [n_atoms, max_neighbors]
        
        # 计算截断函数
        f_cut = cutoff_cosine(dr_norm, self.rc) * topo_mask  # [n_atoms, max_neighbors]
        
        # 计算径向基函数
        exp_term = jnp.exp(-acsf_eta * jnp.square(dr_norm[..., None] - acsf_mus))  # [n_atoms, max_neighbors, n_mu]
        G_raw = exp_term * f_cut[..., None]  # [n_atoms, max_neighbors, n_mu]
        
        # # 按原子类型累积特征
        # type_one_hot = (atype_indices[topo_nblist][..., None] == jnp.arange(self.n_atype))  # [n_atoms, max_neighbors, n_atype]
        
        # # 一次性计算所有特征
        # G = jnp.einsum('ijk,ijl->ikl', G_raw, type_one_hot)  # [n_atoms, n_mu, n_atype]

        # 原代码：按类型one-hot累积（保留n_atype维度）
        # type_one_hot = (atype_indices[topo_nblist][..., None] == jnp.arange(self.n_atype))  # [n_atoms, max_neighbors, n_atype]
        # G = jnp.einsum('ijk,ijl->ikl', G_raw, type_one_hot)  # [n_atoms, n_mu, n_atype]
        
        # 新代码：用嵌入向量加权求和（移除n_atype维度）
        neighbor_types = atype_indices[topo_nblist]  # [n_atoms, max_neighbors]
        # valid_mask = topo_mask.astype(bool)
        # default_type = 0  # 确保该值是有效的原子类型索引（0 <= default_type < n_atype）
        # neighbor_types = jnp.where(valid_mask, neighbor_types, default_type)
        type_embeds = self.atom_embed(neighbor_types)  # [n_atoms, max_neighbors, embed_dim]
        # G_raw: [n_atoms, max_neighbors, n_mu] → 与嵌入向量按邻居维度加权求和
        G = jnp.einsum('ijk,ijl->ikl', G_raw, type_embeds)  # [n_atoms, n_mu, embed_dim]

        return G
    
    def compute_atompair_features(self, cos_gamma_i, cos_gamma_j, j_list, k_list, j_mask, k_mask,
                                  buffer_nblist_inter_rc, atype_indices, apsf_mus, apsf_eta):
        """
        计算原子对对称函数特征
        
        Args:
            cos_gamma_i: i原子的角度余弦值
            cos_gamma_j: j原子的角度余弦值
            j_list: j原子的邻居列表
            k_list: k原子的邻居列表
            j_mask: j原子的掩码
            k_mask: k原子的掩码
            buffer_nblist_inter_rc: 分子间相互作用缓冲
            atype_indices: 原子类型索引
            apsf_mus: APSF mu参数
            apsf_eta: APSF eta参数
        
        Returns:
            jnp.array: 原子对特征
        """
        # 计算 i 和 j 的角度特征
        angle_features_i = jnp.exp(-apsf_eta * jnp.square(cos_gamma_i[..., None] - apsf_mus))
        angle_features_j = jnp.exp(-apsf_eta * jnp.square(cos_gamma_j[..., None] - apsf_mus))

        # # 创建type_one_hot
        # type_one_hot_i = (atype_indices[j_list][..., None] == jnp.arange(self.n_atype))
        # type_one_hot_j = (atype_indices[k_list][..., None] == jnp.arange(self.n_atype))

        # # 应用掩码
        # masked_features_i = angle_features_i * j_mask[..., None]
        # masked_features_j = angle_features_j * k_mask[..., None]

        # # 一次性计算所有类型的贡献
        # G_i = jnp.einsum('ijk,ijl->ikl', masked_features_i, type_one_hot_i)
        # G_j = jnp.einsum('ijk,ijl->ikl', masked_features_j, type_one_hot_j)

        # # 对称平均并应用分子间相互作用掩码
        # G = (G_i + G_j) * 0.5 * buffer_nblist_inter_rc[:, None, None]

        # j_types = atype_indices[j_list]  # [n_pairs, max_neighbors]
        # k_types = atype_indices[k_list]  # [n_pairs, max_neighbors]
        j_types = jnp.take(atype_indices, j_list)  # JAX安全的动态索引
        k_types = jnp.take(atype_indices, k_list)  # JAX安全的动态索引

        # 嵌入层计算
        j_embeds = self.atom_embed(j_types)  # [n_pairs, max_neighbors, embed_dim]
        k_embeds = self.atom_embed(k_types)  # [n_pairs, max_neighbors, embed_dim]

        # 关键：用掩码清零无效位置的嵌入向量
        j_embeds = j_embeds * j_mask[..., None]  # [n_pairs, max_neighbors, embed_dim]
        k_embeds = k_embeds * k_mask[..., None]  # [n_pairs, max_neighbors, embed_dim]
        
                
        # 3. 应用掩码（不变）
        masked_features_i = angle_features_i * j_mask[..., None]  # [n_pairs, max_neighbors, n_mu]
        masked_features_j = angle_features_j * k_mask[..., None]  # [n_pairs, max_neighbors, n_mu]

        # 4. 低维 einsum 操作（内存占用降低 n_atype/embed_dim 倍）
        # 原逻辑：G_i = jnp.einsum('ijk,ijl->ikl', masked_features_i, type_one_hot_i)  # [n_pairs, n_mu, n_atype]
        G_i = jnp.einsum('ijk,ijl->ikl', masked_features_i, j_embeds)  # [n_pairs, n_mu, embed_dim]
        G_j = jnp.einsum('ijk,ijl->ikl', masked_features_j, k_embeds)  # [n_pairs, n_mu, embed_dim]

        # 5. 对称平均（保持逻辑，维度已从n_atype转为embed_dim）
        G = (G_i + G_j) * 0.5 * buffer_nblist_inter_rc[:, None, None]  # [n_pairs, n_mu, embed_dim]

        return G

    def __call__(self, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices):
        """
        特征提取主函数
        
        Args:
            pos: 原子位置
            box: 模拟盒子
            pairs: 原子对索引
            topo_nblist: 拓扑邻居列表
            topo_mask: 拓扑掩码
            mol_ID: 分子ID
            atype_indices: 原子类型索引
        
        Returns:
            tuple: (特征, 距离范数, 缓冲尺度)
        """
        atype_indices = atype_indices - 1 # 关键：1基转0基
        apsf_mus, apsf_eta = self.apsf_mus, self.apsf_eta
        acsf_mus, acsf_eta = self.acsf_mus, self.acsf_eta
        
        # 分子尺度参数
        mScales = jnp.array([0., 0., 0., 0., 0., 1.])
        pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
        nbonds = pairs[:, 2]
        mscales = distribute_scalar(mScales, nbonds - 1)

        pairs = pairs[:, :2]
        buffer_scales = pair_buffer_scales(pairs[:, :2])
        mscales = mscales * buffer_scales

        # 计算原子对距离
        box_inv = jnp.linalg.inv(box)
        ri = pos[pairs[:, 0]]
        rj = pos[pairs[:, 1]]

        rij = rj - ri
        rij = pbc_shift(rij, box, box_inv)
        dr_norm = jnp.linalg.norm(rij, axis=1)

        # 分子内/分子间相互作用处理
        same_mol = mol_ID[pairs[:, 0]] == mol_ID[pairs[:, 1]]
        buffer_inter = jnp.where(same_mol, 0., 1.)
        buffer_intra = jnp.where(same_mol, 1., 0.)
        cutoff = 0.5 * (1 + jnp.cos(jnp.pi * dr_norm / self.rc))
        cutoff = jnp.where(dr_norm <= self.rc, cutoff, 0.0)

        buffer_nblist_inter = buffer_inter * buffer_scales
        buffer_nblist_intra = buffer_intra * buffer_scales
        buffer_nblist_inter_rc = buffer_nblist_inter * cutoff

        # 获取环境原子信息
        j_list, k_list, j_mask, k_mask = get_environment_atoms(pairs, topo_nblist, topo_mask)

        # 计算环境原子的位置和角度（两个方向）
        # i 的环境
        valid_j_mask = j_mask[..., None]
        rj_env = jnp.where(valid_j_mask, pos[j_list], 0.0)
        rj_X = rj_env - ri[:, None, :]
        rj_X = pbc_shift(rj_X, box, box_inv)
        norm_rj_X = jnp.linalg.norm(rj_X, axis=2, keepdims=True) + 1e-10
        rj_X_norm = rj_X / norm_rj_X
        rij_unit = rij / (dr_norm[:, None] + 1e-10)
        cos_gamma_i = jnp.einsum('aji,ai->aj', rj_X_norm, rij_unit) * j_mask

        # j 的环境
        valid_k_mask = k_mask[..., None]
        rk_env = jnp.where(valid_k_mask, pos[k_list], 0.0)
        rk_X = rk_env - rj[:, None, :]
        rk_X = pbc_shift(rk_X, box, box_inv)
        norm_rk_X = jnp.linalg.norm(rk_X, axis=2, keepdims=True) + 1e-10
        rk_X_norm = rk_X / norm_rk_X
        rji_unit = -rij_unit
        cos_gamma_j = jnp.einsum('aji,ai->aj', rk_X_norm, rji_unit) * k_mask

        # 计算原子对特征
        atompair_features = self.compute_atompair_features(
            cos_gamma_i, cos_gamma_j, j_list, k_list, j_mask, k_mask,
            buffer_nblist_inter_rc, atype_indices, apsf_mus, apsf_eta)

        # 计算原子中心特征
        atom_features = self.compute_atomcenter_features(
            pos, box, topo_nblist, topo_mask, atype_indices, acsf_mus, acsf_eta)
        
        atom_features_i = atom_features[pairs[:, 0],]
        atom_features_j = atom_features[pairs[:, 1],]
        atom_features = (atom_features_i + atom_features_j) * 0.5

        # 新逻辑：
        j_type_idx = atype_indices[pairs[:,0]]
        k_type_idx = atype_indices[pairs[:,1]]
        j_embed = self.atom_embed(j_type_idx)  # [n_pairs, embed_dim]
        k_embed = self.atom_embed(k_type_idx)  # [n_pairs, embed_dim]
        atype_features = jnp.concatenate([j_embed, k_embed], axis=1)  # [n_pairs, 2*embed_dim]

        # 合并特征（注意维度变化）
        atom_features = atom_features.reshape(atom_features.shape[0], -1)  # [n_pairs, n_mu*embed_dim]
        atompair_features = atompair_features.reshape(atompair_features.shape[0], -1)  # [n_pairs, n_mu*embed_dim]
        apsf_features = jnp.concatenate((atom_features, atompair_features, atype_features), axis=1)

        # 额外优化：转为float64（内存再减半）
        apsf_features = apsf_features.astype(jnp.float64)
        dr_norm = dr_norm.astype(jnp.float64)

        return apsf_features, dr_norm, buffer_nblist_inter_rc
    
class NeuralNetwork(nn.Module):
    """
    神经网络类
    
    用于从特征预测原子对能量的全连接神经网络
    """
    
    dense_nodes: int = 32  # 隐藏层节点数
    
    @nn.compact
    def __call__(self, combined, dr_norm, buffer_nblist_inter):
        """
        前向传播
        
        Args:
            combined: 组合特征
            dr_norm: 距离范数
            buffer_nblist_inter: 分子间相互作用缓冲
        
        Returns:
            jnp.array: 预测的原子对能量
        """
        x = combined
        for _ in range(1):
            x = nn.Dense(self.dense_nodes)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        out_AB = nn.Dense(1)(x)
        
        return jnp.sum(out_AB * buffer_nblist_inter[:,None])


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    # =============================================================================
    # 参数设置
    # =============================================================================
    parser = argparse.ArgumentParser(description="EAPNN (Equivariant Atom-Pair Neural Network) Training Script")
    parser.add_argument("--rc", type=float, default=6.0, help="Cutoff distance")
    parser.add_argument("--connectivity", type=int, default=4, help="Connectivity level")
    parser.add_argument("--max_neighbors", type=int, default=10, help="Maximum number of neighbors")
    parser.add_argument("--acsf_nmu", type=int, default=20, help="Number of ACSF parameters")
    parser.add_argument("--apsf_nmu", type=int, default=10, help="Number of APSF parameters")
    parser.add_argument("--acsf_eta", type=float, default=100, help="ACSF eta parameter")
    parser.add_argument("--apsf_eta", type=float, default=25, help="APSF eta parameter")
    parser.add_argument("--train_batchsize", type=int, default=64, help="Training batch size")
    parser.add_argument("--test_batchsize", type=int, default=100, help="Testing batch size")
    parser.add_argument("--ff_xml", type=str, default="output.2.ABC.solvents.pospenalty.25.LiNa.AexAes.xml", help="Force field XML file")
    parser.add_argument("--pdb", type=str, default="dimer_000_DEC_DEC_1.pdb", help="PDB file")
    parser.add_argument("--outfile", type=str, default="dataset_eapnn/data_all.xyz", help="Output file")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=1001, help="Number of training epochs")
    parser.add_argument("--dense_nodes", type=int, default=32, help="Number of nodes in dense layers")
    parser.add_argument("--embed_dim", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--use_pbc", action="store_true", help="Use periodic boundary conditions")
    parser.add_argument("--random_seed", type=int, default=1234, help="Random seed for reproducibility")
    parser.add_argument("--save_interval", type=int, default=100, help="Interval for saving model checkpoints")
    parser.add_argument("--eval_interval", type=int, default=10, help="Interval for evaluation and visualization")

    args = parser.parse_args()

    rc = args.rc
    connectivity = args.connectivity
    max_neighbors = args.max_neighbors
    acsf_nmu = args.acsf_nmu
    apsf_nmu = args.apsf_nmu
    acsf_eta = args.acsf_eta
    apsf_eta = args.apsf_eta
    train_batchsize = args.train_batchsize
    test_batchsize = args.test_batchsize
    ff_xml = args.ff_xml
    pdb = args.pdb
    outfile = args.outfile
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    dense_nodes = args.dense_nodes
    embed_dim = args.embed_dim
    use_pbc = args.use_pbc
    random_seed = args.random_seed
    save_interval = args.save_interval
    eval_interval = args.eval_interval

    # =============================================================================
    # 力场和分子设置
    # =============================================================================
    print("正在设置力场和分子...")
    
    # 读取PDB文件
    mol = PDBFile(pdb)
    pos = jnp.array(mol.positions._value) * 10
    box = jnp.array(mol.topology.getPeriodicBoxVectors()._value) * 10

    # 创建哈密顿量
    H = Hamiltonian(ff_xml)
    pots = H.createPotential(mol.topology, nonbondedCutoff=rc*angstrom, 
                           nonbondedMethod=CutoffPeriodic, ethresh=1e-4)


    atype_indices = jnp.array(pots.meta['ADMPPmeForce_map_atomtype'])
    n_atype = len(H.ffinfo['AtomTypes'])

    # 创建邻居列表
    nbl = nblist.NoCutoffNeighborList(pots.meta['cov_map'], padding=True)
    nbl.capacity_multiplier = 1000
    nbl.allocate(pos, box)
    pairs = nbl.pairs

    # 获取分子ID
    mol_ID = []
    for atom in mol.topology.atoms():
        mol_ID.append(atom.residue.index)
    mol_ID = jnp.array(mol_ID)

    # 获取原子元素信息
    # atom_elements = []
    # for atom in mol.topology.atoms():
    #     atom_elements.append(atom.element.atomic_number)
    # z_atomnum = jnp.array(atom_elements)

    # 原子类型处理
    # zindex = [1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 15.0, 16.0]
    # n_atype = len(zindex)
    # z_atomnum_list = [float(num) for num in np.array(z_atomnum)]
    # zindex_dict = {float(num): i for i, num in enumerate(zindex)}
    # atype_indices = jnp.array([zindex_dict.get(num, -1) for num in z_atomnum_list])

    n_atoms = len(pos)
    
    # 获取拓扑邻居
    topo_nblist, topo_mask = get_topology_neighbors(
        pdb, connectivity=connectivity, max_neighbors=max_neighbors, max_n_atoms=None)

    # =============================================================================
    # 模型初始化
    # =============================================================================
    print("正在初始化模型...")
    
    model = EAPNNForce(
        n_atoms=n_atoms, 
        n_atype=n_atype, 
        rc=rc,  
        acsf_nmu=acsf_nmu,
        apsf_nmu=apsf_nmu,
        acsf_eta=acsf_eta,
        apsf_eta=apsf_eta,
        use_pbc=False,
    )

    key = jax.random.PRNGKey(0)
    params_init = model.init(key, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices)
    
    # 测试模型性能
    start_time = time.time()
    params = params_init
    features, dr_norm, buffer_scales = model.apply(
        params, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices, 
        method=model.get_features)
    energy = model.apply(params, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices)
    print(f"初始能量: {energy}")
    end_time = time.time()
    print(f"单次计算耗时: {end_time - start_time:.4f} 秒")

    # 性能测试
    num_runs = 10
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        energy = model.apply(params, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices)
        end_time = time.time()
        total_time += (end_time - start_time)

    average_time = total_time / num_runs
    print(f"平均计算耗时: {average_time:.4f} 秒")

    # =============================================================================
    # 数据加载和预处理
    # =============================================================================
    print("正在加载数据...")
    
    data_ase = outfile
    ase_structures = read(data_ase, ':')
    
    # 构建二聚体类型到PDB路径的映射
    dimer_file_map = {}
    for pdb_path in glob.glob("dimer_bank/*.pdb"):
        filename = os.path.basename(pdb_path)
        parts = filename.split('_')
        monomer_A, monomer_B = parts[-2], parts[-1].split('.')[0]
        dimer_file_map[f"{monomer_A}_{monomer_B}"] = pdb_path
        dimer_file_map[f"{monomer_B}_{monomer_A}"] = pdb_path

    # 填充缓存（在循环外一次性处理所有唯一二聚体）
    unique_dimers = set(structure.info['Comp'].split(':')[0].split('(')[0] + '_' + 
                        structure.info['Comp'].split(':')[1].split('(')[0] 
                        for structure in ase_structures)
    
    print(f"发现 {len(unique_dimers)} 种二聚体类型")
    
    nblist_cache = {}
    for dimer in unique_dimers:
        monomer_A, monomer_B = dimer.split('_')
        if dimer not in dimer_file_map:
            continue
        
        pdb_path = dimer_file_map[dimer]
        mol = PDBFile(pdb_path)
        box = jnp.eye(3) * 50
        H = Hamiltonian(ff_xml)
        pots = H.createPotential(
            mol.topology,
            nonbondedCutoff=rc*angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4
        )
        
        # 计算邻居列表
        pos_dummy = jnp.array(mol.positions._value)
        nbl = nblist.NoCutoffNeighborList(pots.meta['cov_map'], padding=True)
        nbl.capacity_multiplier = 800
        pairs = nbl.allocate(pos_dummy, box)   
             
        # 计算拓扑邻居
        topo_nblist, topo_mask = get_topology_neighbors(
            pdb_path, connectivity=connectivity, max_neighbors=max_neighbors, max_n_atoms=None)
        
        # 存入缓存
        nblist_cache[dimer] = (pairs, topo_nblist, topo_mask)

    # 数据集分析
    print(f"\n数据集分析:")
    print(f"总结构数: {len(ase_structures)}")
    print(f"找到PDB文件的二聚体类型: {len(unique_dimers)}")

    # =============================================================================
    # 数据集分割
    # =============================================================================
    print("正在分割数据集...")
    
    import random
    random.seed(1234)
    random.shuffle(ase_structures)
    train_structures = ase_structures[:int(0.9*len(ase_structures))]
    test_structures = ase_structures[int(0.9*len(ase_structures)):]
    write('test_structures.xyz', test_structures)

    print(f"训练集大小: {len(train_structures)}")
    print(f"测试集大小: {len(test_structures)}")

    # 为训练集添加拓扑信息
    for structure in train_structures:
        comp = structure.info['Comp']
        monomer_A, monomer_B = comp.split(':')
        monomer_A = monomer_A.split('(')[0]
        monomer_B = monomer_B.split('(')[0]
        key = f"{monomer_A}_{monomer_B}"
        
        if key not in nblist_cache:
            raise KeyError(f"缓存缺失二聚体类型: {key}")
        
        pairs, topo_nblist, topo_mask = nblist_cache[key]
        structure.info['pairs'] = pairs
        structure.info['topo_nblist'] = topo_nblist
        structure.info['topo_mask'] = topo_mask

    # 为测试集添加拓扑信息
    for structure in test_structures:
        comp = structure.info['Comp']
        monomer_A, monomer_B = comp.split(':')
        monomer_A = monomer_A.split('(')[0]
        monomer_B = monomer_B.split('(')[0]
        key = f"{monomer_A}_{monomer_B}"
        
        if key not in nblist_cache:
            raise KeyError(f"缓存缺失二聚体类型: {key}")
        
        pairs, topo_nblist, topo_mask = nblist_cache[key]
        structure.info['pairs'] = pairs
        structure.info['topo_nblist'] = topo_nblist
        structure.info['topo_mask'] = topo_mask

    # =============================================================================
    # 数据加载器设置
    # =============================================================================
    print("正在设置数据加载器...")
    
    # 创建数据集和数据加载器
    train_dataset = MoleculeTorchDataset(train_structures, max_neighbors=max_neighbors)
    test_dataset = MoleculeTorchDataset(test_structures, max_neighbors=max_neighbors)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batchsize,
        shuffle=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batchsize,
        shuffle=False,
        drop_last=False
    )

    # 测试数据加载
    try:
        batch = next(iter(train_dataloader))
        print("成功加载批次:")
        for key, value in batch.items():
            print(f"{key}: shape {jnp.array(value).shape}")
    except Exception as e:
        print(f"数据加载错误: {e}")

    # =============================================================================
    # 训练设置
    # =============================================================================
    print("正在设置训练参数...")
    
    # 模型和优化器设置
    key = jax.random.PRNGKey(0)
    model = EAPNNForce(
        n_atoms=n_atoms, 
        n_atype=n_atype, 
        rc=rc,  
        acsf_nmu=acsf_nmu,
        apsf_nmu=apsf_nmu,
        acsf_eta=acsf_eta,
        apsf_eta=apsf_eta,
        use_pbc=False,
    )

    learning_rate = 1e-3
    state = create_train_state(model, learning_rate, key)

    # =============================================================================
    # 可视化设置
    # =============================================================================
    print("正在设置可视化...")
    
    # 设置matplotlib参数
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'DejaVu Sans',
        'mathtext.it': 'DejaVu Sans:italic',
        'mathtext.bf': 'DejaVu Sans:bold',
        'mathtext.default': 'rm',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.max_open_warning': 50,
    })

    # 设置字体
    english_font = fm.FontProperties(family='DejaVu Sans', size=12)
    figure_size = (13, 4)

    # =============================================================================
    # 训练循环
    # =============================================================================
    print("开始训练...")
    
    # 训练参数
    num_epochs = 1001

    # 存储训练历史
    loss_history_train = []
    loss_history_test = []

    start_time = time.time()
    fig = None

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练批次
        train_losses = []
        for batch in train_dataloader:
            batch = torch_batch_to_jax(batch)         
            state, batch_loss = train_step(state, batch)
            train_losses.append(batch_loss)
        
        # 计算epoch平均损失
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{num_epochs} - 平均损失: {avg_train_loss:.4f}")

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
            
        # 每10个epoch评估并可视化
        if epoch % 10 == 0:
            # 记录训练损失
            loss_history_train.append(avg_train_loss)
            
            true_energies = []
            predicted_energies = []
            base_energies = []
            distances = []
            
            # 在测试集上评估模型
            for batch in test_dataloader:
                batch = torch_batch_to_jax(batch)
                pred_energies = jax.vmap(model.apply, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
                    state.params, 
                    batch['pos'], 
                    batch['box'], 
                    batch['pairs'], 
                    batch['topo_nblist'],
                    batch['topo_mask'],
                    batch['molID'],
                    batch['atypes'],
                )
                true_energies.append(batch['energy'] + batch['sr_energy'])
                predicted_energies.append(pred_energies + batch['sr_energy'])
                base_energies.append(batch['sr_energy'])
                distances.append(batch['distance'])

            # 合并数组
            true_energies = np.concatenate(true_energies) 
            predicted_energies = np.concatenate(predicted_energies)
            base_energies = np.concatenate(base_energies)
            distances = np.concatenate(distances)
            
            # 计算误差指标
            rmse = np.sqrt(np.mean((true_energies - predicted_energies)**2))
            rmse_base = np.sqrt(np.mean((true_energies - base_energies)**2))
            mae = np.average(np.absolute(true_energies - predicted_energies))
            loss_history_test.append(mae)

            # 清除之前的输出
            clear_output(wait=True)
            print_training_progress(
                epoch=epoch,
                num_epochs=num_epochs,
                loss=avg_train_loss,
                loss_history_train=loss_history_train,
                loss_history_test=loss_history_test,
                true_energies=true_energies,
                predicted_energies=predicted_energies,
                base_energies=base_energies,
                distances=distances,
                epoch_time=epoch_time,
                total_time=total_time,
                figure_size=figure_size,
                english_font=english_font
            )

            # 保存进度
            if epoch % 100 == 0:
                fig, axs = plot_training_progress(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    loss=avg_train_loss,
                    loss_history_train=loss_history_train,
                    loss_history_test=loss_history_test,
                    true_energies=true_energies,
                    predicted_energies=predicted_energies,
                    base_energies=base_energies,
                    distances=distances,
                    epoch_time=epoch_time,
                    total_time=total_time,
                    figure_size=figure_size,
                    english_font=english_font
                )

                plt.savefig(f'training_progress_epoch_{epoch}.png', dpi=300, bbox_inches='tight', 
                        facecolor='white', pad_inches=0.1)
                final_params = state.params
                with open('model_params.pickle','wb') as ofile:
                    pickle.dump(final_params, ofile)            

    # =============================================================================
    # 训练结束处理
    # =============================================================================
    print("训练完成，正在保存最终模型...")
    
    plt.ioff()  # 关闭交互模式
    if fig is not None:
        plt.close(fig)  # 确保最后一个图形被关闭

    # 最终保存
    final_params = state.params
    with open('final_model_params.pickle', 'wb') as ofile:
        pickle.dump(final_params, ofile)
    
    print(f"训练完成。总耗时: {total_time:.2f} 秒")