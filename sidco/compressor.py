from math import log
import math
from tokenize import String
from numpy import indices
import torch
from torch import Tensor
from bagua.torch_api.tensor import BaguaTensor
import logging


class Compressor():

    iterations = 0
    k_total = 0
    stages = 2
    target_stages = 4 # 
    first_ratio = 0.25
    target_ratio = 0.4
    adaptation_freq = 1
    epsilon_h = 0.1
    epsilon_l = 0.1
    max_stages = 100



    def compress(self, tensor: Tensor, fixed_size = True, ada_stages = True):
        # not sure if this is useful
        #with torch.no_grad():

            def adapt_stages():
                k_avg = self.k_total / self.adaptation_freq
                k = tensor.numel() * self.target_ratio
                cur_stages = self.stages
                if k_avg > k * (1 + Compressor.epsilon_h) and cur_stages > 1:
                    return cur_stages - 1
                elif k_avg < k * (1 + Compressor.epsilon_l) and cur_stages < Compressor.max_stages:
                    return cur_stages + 1
                return cur_stages

            def thresh_estimation_exp(tensor: Tensor, ratio):
                mu = tensor.abs().mean()
                return -mu * log(ratio)

            def apply_threshold(tensor: Tensor, threshold):
                tensor_copy = tensor.clone()
                tensor = tensor.view(-1)
                tensor_copy2 = tensor_copy.view(-1)
                for i in range(len(tensor_copy2)):
                    if tensor_copy2[i] < threshold:
                        tensor_copy2[i] = 0
                return tensor_copy
            
            for i in range(self.stages):
                ratio_per_stage = math.pow(self.target_ratio, 1/self.stages)
                threshold = thresh_estimation_exp(tensor, ratio_per_stage)
                tensor = apply_threshold(tensor, threshold)

            numel = tensor.numel()
            k = math.ceil(self.target_ratio * numel)

            indices = tensor.nonzero().type(torch.int64).to(device='cuda')
            values = torch.tensor([tensor[tuple(index)] for index in indices], device='cuda')

            if ada_stages:
                self.iterations += 1
                
                self.k_total += values.numel()

                if self.iterations % self.adaptation_freq == 0:
                    self.stages = adapt_stages()
                    logging.info("adapted to", self.stages, "stages")
                    self.k_total = 0
            

            if fixed_size:
                k_temp = values.numel()
                if k_temp < k:
                    #print("padding tensor")
                    indsize = list(indices.size())
                    indsize[0] = k
                    indsize = tuple(indsize)
                    new_ind = torch.zeros(indsize, device='cuda', dtype=torch.int64)
                    new_ind[0:k_temp] = indices
                    indices = new_ind

                    valsize = list(values.size())
                    valsize[0] = k
                    valsize = tuple(valsize)
                    new_val = torch.zeros(valsize, device='cuda')
                    new_val[0:k_temp] = values
                    values = new_val

                if k_temp > k:
                    # TODO take random elements instead of the first k
                    #print("shortening tensor")
                    indices = indices[0:k]
                    values = values[0:k]


            return indices, values

    def decompress(self, size: Tensor, indices: Tensor, values: Tensor, out:Tensor =None):
        #with torch.no_grad():
            if out is None:
                out = torch.zeros(size=[d for d in size])

            for index, value in zip(indices, values):
                out[tuple(index)] = value

            return out