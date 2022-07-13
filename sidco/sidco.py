#!/usr/bin/env python3

from time import sleep
from numpy import dtype, indices
from bagua.torch_api import communication
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms.base import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from bagua.torch_api.tensor import BaguaTensor
from typing import List

import torch
from torch import Tensor, tensor
import logging

from compressor import Compressor



class SidcoImpl(AlgorithmImpl):
    compressors = {}

    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = False,
        average: bool = True,
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

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel) -> List[BaguaTensor]:
        tensors = super().init_tensors(bagua_ddp)
        for t in tensors:
            self.compressors[t.bagua_tensor_name] = Compressor()
        return tensors


    # One tensor into one bucket for now
    # TODO: remove this and make the init_operations() work with multiple tensors per bucket
    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        bagua_buckets = []
        for id1, tensorlist in enumerate(tensors):
            for id2, tensor in enumerate(tensorlist):
                bagua_bucket = BaguaBucket(tensors=[tensor], name=str(id1) + "-" +str(id2), flatten=True)
                bagua_buckets.append(bagua_bucket)
        return bagua_buckets


    def init_operations(
        self,
        bagua_ddp: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()

        # for some reason without this the cuda memory quickly fills up
        for bucket in bagua_ddp.bagua_buckets:
            pass

        def allgather(*args):
            nranks = len(bagua_ddp.process_group.ranks)


            for btensor in bucket.tensors:

                name = btensor.bagua_tensor_name
                tensor = btensor.bagua_getter_closure()

                logging.debug("compressing tensor {} ...".format(name))
                indices, values = self.compressors[name].compress(tensor, ada_stages=False) #TODO adapt stages
                
                logging.debug("gathering indices and values...")
                recv_ind = []
                for t in indices:
                    all_t = torch.zeros((nranks,) + indices[0].size(), device='cuda', dtype=torch.int64)
                    communication.gather(t, all_t, 0)
                    recv_ind.append(all_t)
                
                recv_val = torch.zeros(((nranks,) + values.size()), device='cuda')
                communication.gather(values, recv_val, 0)

                logging.debug("creating and filling new tensor")
                new_tensor = torch.zeros_like(tensor)
                counter = torch.zeros_like(tensor)
                for i in range(nranks):
                    ind = ()
                    for recv_ind_dim in recv_ind:
                        ind = ind + (recv_ind_dim[i],)
                    new_tensor[ind] += recv_val[i]
                    counter[ind] += 1
                
                counter = counter + (counter == 0)
                new_tensor.div_(counter)

                logging.debug("recompressing tensor for broadcasting...")
                indices, values = self.compressors[name].compress(new_tensor, ada_stages=False)

                logging.debug("broadcasting indices and values...")
                new_indices = ()
                for t in indices:
                    communication.broadcast(t, 0)
                    new_indices = new_indices + (t,)
                indices = new_indices
                communication.broadcast(values, 0)

                logging.debug("decompressing tensor...")
                tensor = self.compressors[name].decompress(indices, values, tensor)

                btensor.bagua_setter_closure(tensor)

                #sleep(0.5) #TODO remove!

        bucket.append_python_op(allgather)



        # tensor_sizes = list(map(lambda t : t.bagua_getter_closure().size(), bucket.tensors))
        # tensor_names = list(map(lambda t : t.bagua_tensor_name, bucket.tensors))

        # def compress(*args):
        #     #print("name", bucket.tensors[0].bagua_tensor_name)
        #     compressed_tensors = list()
        #     for btensor in bucket.tensors:
        #         tensor = btensor.bagua_getter_closure()
        #         name = btensor.bagua_tensor_name
        #         logging.info("compressing tensor", name, "...")
        #         indices, values = self.compressors[btensor.bagua_tensor_name].compress(tensor)
        #         indices = BaguaTensor.ensure_bagua_tensor(indices, name=f"{name}_indices")
        #         values = BaguaTensor.ensure_bagua_tensor(values, name=f"{name}_values")
        #         compressed_tensors.append(indices)
        #         compressed_tensors.append(values)
        #     bucket.tensors = compressed_tensors

        # def decompress(*args):
        #     logging.info("decompressing tensors...")
        #     new_tensors = list()
        #     tensors = bucket.tensors
        #     for i in range(0, int(len(tensors)/2)):
        #         size = tensor_sizes[i]
        #         name = tensor_names[i]
        #         indices = bucket.tensors[2*i].bagua_getter_closure()
        #         values = bucket.tensors[2*i+1].bagua_getter_closure()
        #         tensor = self.compressors[name].decompress(size, indices, values)
        #         baguat = BaguaTensor.ensure_bagua_tensor(tensor, name=name)
        #         new_tensors.append(baguat)
        #     bucket.tensors = new_tensors


        # bucket.append_python_op(compress)
        # bucket.append_centralized_synchronous_op(
        #     hierarchical=self.hierarchical,
        #     average=self.average,
        #     group=self.process_group,
        # )
        # bucket.append_python_op(decompress)



class Sidco(Algorithm):
    def __init__(self, hierarchical: bool = False, average: bool = True):
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

    def reify(self, process_group: BaguaProcessGroup) -> SidcoImpl:
        return SidcoImpl(
            process_group,
            hierarchical=self.hierarchical,
            average=self.average,
        )
