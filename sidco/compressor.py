from math import log
import math
from tokenize import String
from numpy import indices
import torch
from torch import Tensor
from bagua.torch_api.tensor import BaguaTensor
import logging
import time


class Compressor():

    iterations = 0
    k_total = 0
    stages = 2
    target_stages = 4 # 
    first_ratio = 0.25
    adaptation_freq = 1
    epsilon_h = 0.1
    epsilon_l = 0.1
    max_stages = 100

    
    def __init__(self, ratio: float = 0.1) -> None:
        self.target_ratio = ratio


    def compress(
        self, 
        tensor: Tensor, 
        fixed_size = True, 
        ada_stages = True, 
        indices: Tensor = None, 
        values: Tensor = None,
        ):

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
                tensor_copy2 = tensor_copy.view(-1)
                abs_tensor_copy = tensor.clone().abs().view(-1)
                ones = abs_tensor_copy > threshold
                tensor_copy2.mul_(ones)
                return tensor_copy
            
            start = time.time()

            logging.debug("Tensor of size {}:".format(tensor.size()))
            for i in range(self.stages):
                ratio_per_stage = math.pow(self.target_ratio, 1/self.stages)
                thresh_start = time.time()
                threshold = thresh_estimation_exp(tensor, ratio_per_stage)
                thresh_end = time.time()
                logging.debug("{:.4f}: thresh_estimation_exp()".format(thresh_end - thresh_start))
                tensor = apply_threshold(tensor, threshold)
                thresh_apply = time.time()
                logging.debug("{:.4f}: apply_threshold()".format(thresh_apply-thresh_end))

            numel = tensor.numel()
            k = math.ceil(self.target_ratio * numel)

            start_indices = time.time()
            indices = tensor.nonzero(as_tuple=True)
            end_indices = time.time()
            logging.debug("{:.4f}: creating index tensor".format(end_indices-start_indices))
            values = tensor[indices]
            end_values = time.time()
            logging.debug("{:.4f}: creating value tensor".format(end_values-end_indices))


            if ada_stages:
                start_ada = time.time()
                self.iterations += 1
                
                self.k_total += values.numel()

                if self.iterations % self.adaptation_freq == 0:
                    self.stages = adapt_stages()
                    logging.info("adapted to {} stages".format(self.stages))
                    self.k_total = 0
                end_ada = time.time()
                logging.debug("{:.4f}: Stage adaptation".format(end_ada-start_ada))

            if fixed_size:
                start_adjust = time.time()
                k_temp = values.numel()
                if k_temp < k:
                    #print("padding tensor")
                    new_indices = ()
                    for t in indices:
                        new_t = torch.zeros(k, device='cuda', dtype=torch.int64)
                        new_t[0:k_temp] = t
                        new_indices = new_indices + (new_t,)
                    indices = new_indices

                    new_val = torch.zeros(k, device='cuda')
                    new_val[0:k_temp] = values
                    values = new_val

                    end_padding = time.time()
                    logging.debug("{:.4f}: padding".format(end_padding-start_adjust))

                if k_temp > k:
                    # TODO take random elements instead of the first k
                    #print("shortening tensor")
                    new_indices = ()
                    for t in indices:
                        new_t = t[0:k]
                        new_indices = new_indices + (new_t,)
                    indices = new_indices

                    values = values[0:k]
                    end_shortening = time.time()
                    logging.debug("{:.4f}: shortening".format(end_shortening-start_adjust))

            end = time.time()
            logging.debug("{:.4f}: Total compression time".format(end-start))

            return indices, values

    def decompress(self, indices: Tensor, values: Tensor, out: Tensor = None, size: torch.Size = None):
        #with torch.no_grad():
            start = time.time()
            if out is None:
                if size is None:
                    raise Exception("Either size or out tensor must be given")
                out = torch.zeros(size=[d for d in size], device='cuda')
            out[indices] = values
            end = time.time()
            logging.debug("{:.4f}: Total decompression time".format(end-start))

            return out