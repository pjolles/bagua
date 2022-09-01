#!/usr/bin/env python3

import math
from time import sleep
import time
from bagua.torch_api import communication
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms.base import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from bagua.torch_api.tensor import BaguaTensor
from typing import List

import torch
import logging

from compressor import Compressor



class SidcoImpl(AlgorithmImpl):
    compressors = {}

    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = False,
        average: bool = True,
        ratio: float = 0.1
    ):
        """
        Implementation of the
        `GradientAllReduce <https://tutorials.baguasys.com/algorithms/gradient-allreduce>`_
        algorithm.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            hierarchical (bool): Enable hierarchical communication.
            average (bool): If ``True``, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        super(SidcoImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.average = average
        self.ratio = ratio

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel) -> List[BaguaTensor]:
        tensors = super().init_tensors(bagua_ddp)
        for t in tensors:
            self.compressors[t.bagua_tensor_name] = Compressor(ratio=self.ratio)
        return tensors


    # One tensor into one bucket for now
    # def tensors_to_buckets(
    #     self, tensors: List[List[BaguaTensor]], do_flatten: bool
    # ) -> List[BaguaBucket]:
    #     bagua_buckets = []
    #     for id1, tensorlist in enumerate(tensors):
    #         for id2, tensor in enumerate(tensorlist):
    #             bagua_bucket = BaguaBucket(tensors=[tensor], name=str(id1) + "-" +str(id2), flatten=True)
    #             bagua_buckets.append(bagua_bucket)
    #     return bagua_buckets


    def init_operations(
        self,
        bagua_ddp: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        nranks = len(bagua_ddp.process_group.ranks)

        # Initialize the tensors to be exchanged
        residuals = {}
        indices = {}
        values = {}
        recv_indices = {}
        recv_values = {}
        for btensor in bucket.tensors:
            name = btensor.bagua_tensor_name
            t = btensor.bagua_getter_closure()
            residuals[name] = torch.zeros_like(t.view(-1))
            indices[name] = ()
            recv_indices[name] = ()
            indices[name] = torch.zeros(math.ceil(self.ratio * t.numel()), 
                                        device='cuda', 
                                        dtype=torch.int64)
            recv_indices[name] = torch.zeros(nranks, math.ceil(self.ratio * t.numel()), 
                                        device='cuda',
                                        dtype=torch.int64)
            values[name] = torch.zeros(math.ceil(self.ratio * t.numel()), 
                                        device='cuda')
            recv_values[name] = torch.zeros(nranks, math.ceil(self.ratio * t.numel()), 
                                        device='cuda')

        def allgather(*args):

            for btensor in bucket.tensors:

                time_0 = time.time()
                name = btensor.bagua_tensor_name
                tensor = btensor.bagua_getter_closure()

                logging.debug("Tensor name: {}, size: {}".format(name, tensor.size()))

                #logging.debug("compressing tensor {} ...".format(name))
                time_1 = time.time()
                ind, val, residuals[name] = self.compressors[name].compress(
                    tensor, 
                    fixed_size = True,
                    ada_stages = True,
                    residual = residuals[name]
                ) #TODO adapt stages

                indices[name].copy_(ind)
                values[name].copy_(val)

                #logging.debug("gathering indices and values...")
                time_2 = time.time()
                communication.gather(indices[name], recv_indices[name], 0)
                communication.gather(values[name], recv_values[name], 0)
                time_3 = time.time()

                logging.debug("{:.4f}: gather".format(time_3 - time_2))

                #logging.debug("decompressing tensors")
                tensor.zero_()
                for i in range(nranks):
                    tensor.add_(self.compressors[name].decompress(
                        recv_indices[name][i],
                        recv_values[name][i],
                        size = tensor.size()
                    ))

                tensor.div_(nranks)
                
                #logging.debug("recompressing tensor for broadcasting...")
                ind, val, residual = self.compressors[name].compress(
                    tensor,
                    fixed_size = True,
                    ada_stages = False,
                    residual = residuals[name]
                )
                indices[name].copy_(ind)
                values[name].copy_(val)

                #logging.debug("broadcasting indices and values...")
                time_4 = time.time()
                communication.broadcast(indices[name], 0)
                communication.broadcast(values[name], 0)
                time_5 = time.time()

                logging.debug("{:.4f}: broadcast".format(time_5 - time_4))

                #logging.debug("decompressing tensor...")
                tensor = self.compressors[name].decompress(
                    indices[name], 
                    values[name], 
                    size=tensor.size()
                )

                btensor.bagua_setter_closure(tensor)
                time_6 = time.time()
                logging.debug("{:.4f}: Total communication time".format(time_6 - time_0))

        bucket.append_python_op(allgather)

        def test(*args):
            for btensor in bucket.tensors:
                tensor = btensor.bagua_getter_closure()
                ind, val = self.compressors[btensor.bagua_tensor_name].compress(tensor, ada_stages=False)
                tensor = self.compressors[btensor.bagua_tensor_name].decompress(ind, val, size=tensor.size())
                btensor.bagua_setter_closure(tensor)

        #bucket.append_python_op(test)


class Sidco(Algorithm):
    def __init__(self, hierarchical: bool = False, average: bool = True, ratio: float = 0.1):
        """
        Create an instance of the
        `GradientAllReduce <https://tutorials.baguasys.com/algorithms/gradient-allreduce>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            average (bool): If ``True``, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        self.hierarchical = hierarchical
        self.average = average
        self.ratio = ratio

    def reify(self, process_group: BaguaProcessGroup) -> SidcoImpl:
        return SidcoImpl(
            process_group,
            hierarchical=self.hierarchical,
            average=self.average,
            ratio=self.ratio
        )
