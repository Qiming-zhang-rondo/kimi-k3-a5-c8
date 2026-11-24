#   Copyright (c) 2026 Huawei Technologies Co., Ltd.
#   This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#   CANN Open Software License Agreement Version 2.0 (the "License").
#   Please refer to the License for details. You may not use this file except in compliance with the License.
#   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#   INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#   See LICENSE in the root of the software repository for the full text of the License.

from typing import Any

import numpy as np
import torch
import torch_npu
from torch.autograd import Function
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

import cloud_ops_turbo_lib


class CloudSparseConv2dFunction(Function):
    @staticmethod
    def forward(
        ctx: Any,
        features,
        indices,
        weight,
        out_spatial_shape,
        out_channels,
        batch_size,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    ) -> torch.Tensor:

        device = features.device
        weight = weight.data

        outidx_pair, ouidx_offset = cloud_ops_turbo_lib.cloud_sparse_conv2d(
            indices, kernel_size, stride, padding, out_channels, out_spatial_shape, batch_size
        )

        # sort and nonezero
        num_voxels_, uni_voxels, unique_indices_offset, sorted_idx_to_former_indices, uni_argsort_indices = (
            cloud_ops_turbo_lib.cloud_unique_voxel(ouidx_offset)
        )
        indices_last = torch.tensor(ouidx_offset.shape).to(unique_indices_offset.device)
        unique_indices_offset = torch.cat((unique_indices_offset, indices_last), dim=0)

        # index_put and matmul
        out_features, outidx = cloud_ops_turbo_lib.cloud_to_sparse2d(
            features, weight, unique_indices_offset.int(), sorted_idx_to_former_indices.int(), outidx_pair.int()
        )

        outidx = outidx[:, 0:3]

        ctx.save_for_backward(features, weight, sorted_idx_to_former_indices.int(), unique_indices_offset.int())
        return out_features, outidx


def generate_map2d(coors, origin_spatial_shape, bs, _kernel_size):
    spatial_shape = (
        origin_spatial_shape[0] + 2 * (_kernel_size[0] // 2),
        origin_spatial_shape[1] + 2 * (_kernel_size[1] // 2),
    )
    _padding = _kernel_size[0] // 2
    spatial_shape_size = spatial_shape[0] * spatial_shape[1]
    sparse_rate = coors.shape[0] / (spatial_shape_size * bs)
    flatten_indices = coors[:, 0] * spatial_shape_size + coors[:, 1] * (spatial_shape[1]) + coors[:, 2]

    flatten_indices += (spatial_shape[1] + 1) * _padding
    map1 = torch.full((spatial_shape_size * bs,), -1, dtype=torch.int32, device=coors.device)

    map1[flatten_indices] = torch.arange(flatten_indices.numel(), dtype=torch.int32, device=coors.device)
    map2 = torch.Tensor([]).int()
    # [batch, h, w] -> [batch, h , w, 0]
    coors = F.pad(coors, (0, 1))
    return map1, map2, spatial_shape, coors, sparse_rate


def generate_map2d_arch35(coors, spaned_spatial_shape, bs, kernel_size):
    padding = kernel_size[0] // 2
    spatial_shape_size = spaned_spatial_shape[0] * spaned_spatial_shape[1]
    
    if spatial_shape_size > 400000000:
        spatial_shape1 = spaned_spatial_shape[0]
        new_coors1 = (
            spatial_shape1 * coors[:, 0]
            + coors[:, 1]
            + padding
        )

        map1 = torch.full((spatial_shape1 * bs,), -1, dtype=torch.int32, device=coors.device)
        map1_length, unique_idx, _, _, _ = cloud_ops_turbo_lib.cloud_unique_voxel(new_coors1)
        map1[unique_idx] = torch.arange(map1_length, dtype=torch.int32, device=coors.device)
        map2 = torch.full((map1_length, spaned_spatial_shape[1]), -1, dtype=torch.int32, device=coors.device)
        map2[map1[new_coors1], (coors[:, 2] + padding)] = torch.arange(new_coors1.numel(), dtype=torch.int32, device=coors.device)
    else:
        map1 = torch.full((bs, spaned_spatial_shape[0], spaned_spatial_shape[1]), -1,
            dtype=torch.int32, device=coors.device)
        bs, sp0, sp1 = coors.split(1, dim=1)
        bs, sp0, sp1 = bs.flatten(), sp0.flatten(), sp1.flatten()
        map1[bs, sp0 + padding, sp1 + padding] = torch.arange(coors.shape[0], dtype=torch.int32, device=coors.device)
        map1 = map1.flatten().contiguous()
        map2 = torch.Tensor([]).int().to(coors.device)
    
    return map1, map2


class SubMConv2dFunction(Function):
    @staticmethod
    def forward(
        ctx: Any,
        features,
        indices,
        weight,
        out_spatial_shape,
        out_channels,
        batch_size,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        indices_offset,
    ) -> torch.Tensor:
        weight = weight.data
        indices = indices.contiguous()
        device = indices.device

        DEVICE_NAME = torch_npu.npu.get_device_name(features.device.index)
        if 'Ascend950' in DEVICE_NAME:
            spaned_spatial_shape = (out_spatial_shape[0] + 2 * (kernel_size[0] // 2), out_spatial_shape[1] + 2 * (kernel_size[1] // 2))
            if indices_offset is None:
                map1, map2 = generate_map2d_arch35(indices, spaned_spatial_shape, batch_size, kernel_size)
                with_key = 0
                indices_offset = torch.Tensor([]).int().to(device)
            else:
                map1, map2 = torch.Tensor([]).int().to(device), torch.Tensor([]).int().to(device)
                with_key = 1

            out_features, out_indices_offset = cloud_ops_turbo_lib.cloud_subm_sparse_conv2d_with_matmul(features, weight, indices, indices_offset, map1, map2, kernel_size,
                features.shape[1], out_channels, spaned_spatial_shape, batch_size, with_key)

            indices_offset = indices_offset if with_key == 1 else out_indices_offset
            ctx.save_for_backward(features, weight, indices_offset)
        elif 'Ascend910B' in DEVICE_NAME or 'Ascend910_93' in DEVICE_NAME:
            map1, map2, spaned_spatial_shape, new_indices, sparse_rate = generate_map2d(
                indices, out_spatial_shape, batch_size, kernel_size
            )
            output_iml2col, indices_offset = cloud_ops_turbo_lib.cloud_subm_sparse_conv2d(
                features,
                new_indices,
                map1.npu(),
                map2.npu(),
                kernel_size,
                features.shape[1],
                spaned_spatial_shape,
                batch_size,
                sparse_rate,
            )
            out_features = output_iml2col @ weight.reshape(-1, out_channels)
            ctx.kernel_size = kernel_size
            ctx.save_for_backward(features, weight, output_iml2col, indices_offset)
        else:
            raise NotImplementedError(
                'The CloudSubMConv2d operator currently only supports Ascend910B, Ascend910C and Ascend950.'
            )
        return out_features, indices, indices_offset


class SparseInverseConv3dFunction(Function):
    @staticmethod
    def forward(
        ctx: Any,
        features,
        weight,
        in_channels,
        out_channels,
        kernel_size,
        indicedata,
    ) -> torch.Tensor:
        weight = weight.data
        origin_indices = indicedata.origin_indices
        unique_indices_offset = indicedata.unique_indices_offset.int()
        sorted_idx_to_former_indices = indicedata.sorted_idx_to_former_indices

        DEVICE_NAME = torch_npu.npu.get_device_name(features.device.index)
        if 'Ascend950' in DEVICE_NAME:
            out_features = cloud_ops_turbo_lib.cloud_sparse_inverse_conv3d_with_matmul(
                features,
                weight,
                origin_indices,
                unique_indices_offset,
                sorted_idx_to_former_indices,
            )
            ctx.save_for_backward(
                features,
                weight,
                unique_indices_offset,
                sorted_idx_to_former_indices,
                indicedata.outidx_pair,
            )
        elif 'Ascend910' in DEVICE_NAME:
            output_img2col = cloud_ops_turbo_lib.cloud_sparse_inverse_conv3d(
                features,
                origin_indices,
                unique_indices_offset,
                sorted_idx_to_former_indices,
                kernel_size,
                in_channels,
            )
            out_features = output_img2col @ weight.reshape(-1, out_channels)
            ctx.save_for_backward(
                weight,
                unique_indices_offset,
                sorted_idx_to_former_indices,
                indicedata.outidx_pair,
                output_img2col,
            )
        return out_features

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_out_features: torch.Tensor) -> tuple:
        DEVICE_NAME = torch_npu.npu.get_device_name(grad_out_features.device.index)
        if 'Ascend950' in DEVICE_NAME:
            features, weight, unique_indices_offset, sorted_idx_to_former_indices, outidx_pair = ctx.saved_tensors
            weight_shape = weight.shape
            inverse_feature_grad, inverse_weight_grad = cloud_ops_turbo_lib.cloud_sparse_inverse_conv3d_grad(
                features,
                weight,
                grad_out_features,
                unique_indices_offset,
                sorted_idx_to_former_indices,
            )
            inverse_weight_grad = inverse_weight_grad.view(
                weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3], weight_shape[4]
            )
        elif 'Ascend910' in DEVICE_NAME:
            weight, unique_indices_offset, sorted_idx_to_former_indices, outidx_pair, output_img2cols = (
                ctx.saved_tensors
            )
            weight_shape = weight.shape
            weight_permuted = weight.permute(0, 1, 2, 4, 3).contiguous()

            inverse_feature_grad, _ = cloud_ops_turbo_lib.cloud_sparse_matmul(
                grad_out_features, weight_permuted, unique_indices_offset, sorted_idx_to_former_indices, outidx_pair
            )
            inverse_weight_grad = grad_out_features.transpose(0, 1).contiguous() @ output_img2cols

            inverse_weight_grad = (
                inverse_weight_grad.transpose(0, 1)
                .contiguous()
                .view(weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3], weight_shape[4])
            )

        return inverse_feature_grad, inverse_weight_grad, None, None, None, None


indice_conv2d = CloudSparseConv2dFunction.apply
indice_subm_conv2d = SubMConv2dFunction.apply
indice_inverse_conv3d = SparseInverseConv3dFunction.apply
