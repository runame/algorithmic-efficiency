"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of the following repository:
https://github.com/facebookresearch/optimizers/

"""
# pylint: disable=invalid-name
from copy import deepcopy
import enum
import logging
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.autograd import profiler
import torch.distributed as dist

try:
  # DTensor requires PyTorch 2.1 nightly build.
  from torch.distributed._tensor import zeros as dtensor_zeros
  import torch.distributed._tensor as dtensor

  # Flag that DTensor is enabled.
  ENABLE_DTENSOR = True

  # Cache for device meshes for allocating distributed tensors.
  _device_mesh_cache: Dict[str, dtensor.DeviceMesh] = {}

except ImportError:
  # If we encounter an import error, turns off DTensor.
  ENABLE_DTENSOR = False

ALIGNMENT_BYTES = (
    64  # necessary for determining buffer size, possibly hardware-dependent
)

logger: logging.Logger = logging.getLogger(__name__)

if not ENABLE_DTENSOR:
  logger.warning(
      "DTensor is not available and was not imported. Continuing with Tensor..."
  )

# TODO: Support additional data structures
COMPATIBLE_DATA_STRUCTURES = (list, tuple, set)
ALL_CLASSES = (Tensor, dict) + COMPATIBLE_DATA_STRUCTURES


def are_states_equal(prev_state: Any, new_state: Any) -> bool:
  r"""
    Comparison function that checks whether or not two nested state dictionaries
    containing tensors or other custom data types are equal.

    Useful for debugging purposes.

    Args:
        prev_state (Any): State to compare.
        new_state (Any): State to compare.

    """

  if not isinstance(new_state, type(prev_state)):
    return False

  if isinstance(prev_state, Tensor):
    return torch.equal(prev_state, new_state)
  elif isinstance(prev_state, dict):
    prev_keys = prev_state.keys()
    if prev_keys != new_state.keys():
      return False
    return all(
        are_states_equal(prev_state[key], new_state[key]) for key in prev_keys)
  else:
    return prev_state == new_state


class OptimizerModule:
  r"""
    Optimizer module that supports state_dict and load_state_dict functions that
    recursively constructs the state dictionary by examining other
    OptimizerModule objects. Similar to nn.Module but "trims the fat" by
    removing unnecessary functions for more general optimizer modules.

    When generating the state_dict, looks at the internal dictionary and
    recursively calls state_dict on other optimizer modules.

    """

  def _save_to_state_dict(
      self,
      states: Iterable,
      destination: Dict,
      keep_vars: bool,
      store_non_tensors: bool,
  ):
    r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~OptimizerModule.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            states (Iterable): iterable that gives tuples of values to be stored
                in destination dict
            destination (dict): a dict where state will be stored
            keep_vars (bool): keep variables for tensor
            store_non_tensors (bool): flag for storing non-tensor objects

        """

    for key, value in states:
      # TODO: Add case for ShardedTensor
      if isinstance(value, Tensor):
        destination[key] = value if keep_vars else value.detach()
      elif isinstance(value, OptimizerModule):
        destination[key] = {}
        value.state_dict(
            destination=destination[key],
            keep_vars=keep_vars,
            store_non_tensors=store_non_tensors,
        )
      elif isinstance(value, dict):
        destination[key] = {}
        self._save_to_state_dict(
            states=value.items(),
            destination=destination[key],
            keep_vars=keep_vars,
            store_non_tensors=store_non_tensors,
        )
      elif isinstance(value, COMPATIBLE_DATA_STRUCTURES):
        destination[key] = {}
        self._save_to_state_dict(
            states=enumerate(value),
            destination=destination[key],
            keep_vars=keep_vars,
            store_non_tensors=store_non_tensors,
        )
      elif store_non_tensors:
        destination[key] = value

  def state_dict(
      self,
      destination: Optional[Dict] = None,
      keep_vars: bool = False,
      store_non_tensors: bool = False,
  ) -> Dict[str, Any]:
    r"""Returns a nested state dictionary containing a whole internal
        dict of the module. OptimizerModules and other common data structures
        are represented by a dictionary within the dict.

        .. warning::
            Please avoid the use of argument ``destination`` as it is not
            designed for end-users.

        Args:
            destination (dict, optional): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            keep_vars (bool, optional): by default the :class:`~Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.
            store_non_tensors (bool, optional): flag for storing non-tensor
                objects. Default: ``False``.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        """

    if destination is None:
      destination = {}

    self._save_to_state_dict(self.__dict__.items(),
                             destination,
                             keep_vars,
                             store_non_tensors)

    return destination

  def _load_from_state_dict(self,
                            old_state: Any,
                            new_state: Any,
                            store_non_tensors: bool) -> Any:
    if isinstance(old_state, Tensor):
      if not isinstance(new_state, Tensor):
        logger.warning(
            f"Both old state {old_state} and new state {new_state} must be "
            "tensors! Continuing...")
        return old_state
      old_state.detach().copy_(new_state)
    elif isinstance(old_state, OptimizerModule):
      old_state.load_state_dict(new_state, store_non_tensors)
    elif isinstance(old_state, dict):
      if not isinstance(new_state, dict):
        logger.warning(
            f"Both old state {old_state} and new_state {new_state} must be "
            "dicts! Continuing...")
        return old_state
      for key, old_value in old_state.items():
        if key in new_state:
          old_state[key] = self._load_from_state_dict(
              old_state=old_value,
              new_state=new_state[key],
              store_non_tensors=store_non_tensors,
          )
    elif isinstance(old_state, COMPATIBLE_DATA_STRUCTURES):
      old_state = type(old_state)(
          self._load_from_state_dict(
              old_state=old_value,
              new_state=new_state[i],
              store_non_tensors=store_non_tensors,
          ) if store_non_tensors or
          isinstance(old_value, ALL_CLASSES + (OptimizerModule,)) else old_value
          for i,
          old_value in enumerate(old_state))
    elif store_non_tensors:
      if not isinstance(new_state, type(old_state)):
        logger.warning(f"Types of old value {type(old_state)} and new value "
                       f"{type(new_state)} do not match! Continuing...")
        return old_state
      old_state = deepcopy(new_state)

    return old_state

  def load_state_dict(self,
                      state_dict: Mapping[str, Any],
                      store_non_tensors: bool = False) -> None:
    """
        This implementation requires the stored and loaded states to be fully
        initialized.

        Because of introduced strictness it allows us to:
            * do compatibility checks for state and param_groups, which improves
              usability
            * avoid state duplication by directly copying into state tensors,
              e.g. optimizer.step()  # make sure optimizer is initialized
              sd = optimizer.state_dict()
              load_checkpoint(sd)  # copy state directly into tensors, re-shard
              if needed optimizer.load_state_dict(sd)  # replace param_groups

        Args:
            state_dict (dict): State dictionary to load
            store_non_tensors (bool, optional): Load non-tensor objects

        """

    # load state
    self._load_from_state_dict(self.__dict__, state_dict, store_non_tensors)


def _build_full_key(key: str, key_prefix: Optional[str] = None) -> str:
  return f"{key_prefix}.{key}" if key_prefix is not None else key


def _flatten(
    input_dict: Dict[str, Any],
    output_dict: Dict[str, Any],
    key_prefix: Optional[str] = None,
) -> None:
  """Recursive flattening function for checkpointing support.

    Args:
        input_dict (Dict[str, Any]): Input dictionary to flatten.
        output_dict (Dict[str, Any]): Flattened dictionary.
        key_prefix (str): Optional prefix for flattening. (Default: None)

    """
  for k, v in input_dict.items():
    key = _build_full_key(k, key_prefix)
    if key in output_dict:
      raise KeyError(
          f"{key} already exists in output. Overwriting is not allowed. "
          f"If {k} is desired at this level, please consider updating "
          f"parent level keys if possible as a workaround.")
    if isinstance(v, dict):
      _flatten(v, output_dict, key_prefix=key)
      continue
    output_dict[key] = v


def flatten_state_dict(
    input_dict: Dict[str, Any],
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
  """Flattens state dictionary.

    Used for supporting distributed checkpointing solution.

    Args:
        input_dict (Dict[str, Any]): Input dictionary to flatten.
        prefix (str): Optional prefix for dictionary. (Default: None)

    Returns:
        output_dict (Dict[str, Any]): Flattened dictionary.

    """
  output_dict: Dict[str, Any] = {}
  _flatten(input_dict, output_dict, prefix)
  return output_dict


def distribute_buffer_sizes(buffer_sizes: List[int],
                            group_size: int) -> List[Tuple[int, int]]:
  """Distribute given buffer sizes across ranks in a group.

    Buffer sizes will be rounded up for memory allocation. Buffers are
    distributed such that total buffer sizes of each rank are as even as
    possible. This is currently performed using a greedy algorithm. We do not
    currently consider computational cost or kernel launching overheads.

    TODO: Explore a better distribution strategy.

    Args:
        buffer_sizes (List[int]): buffer sizes
        group_size (int): the size of groups.

    Returns:
        buffer_size_ranks (List[Tuple[int, int]]): a list of pairs of buffer
        size and an assigned rank.

    Example:
        buffer_sizes = [128, 64, 500, 256], group_size = 2
        -> buffer_size_ranks = [(128, 1), (64, 1), (512, 0), (256, 1)]

    """

  # Allocate them greedily (note: Python's "sorted" is stable)
  buffer_size_ranks = [(-1, -1)] * len(buffer_sizes)
  buffer_size_sums = [0] * group_size
  for index, buffer_size in sorted(
      enumerate(buffer_sizes),
      key=lambda t: t[1],
      reverse=True,
  ):
    # computes smallest multiple of ALIGNMENT_BYTES that is >= buffer size
    aligned_buffer_size = ((buffer_size + ALIGNMENT_BYTES - 1) //
                           ALIGNMENT_BYTES * ALIGNMENT_BYTES)
    rank = buffer_size_sums.index(min(buffer_size_sums))
    buffer_size_sums[rank] += aligned_buffer_size
    buffer_size_ranks[index] = (aligned_buffer_size, rank)

  return buffer_size_ranks


def split_local_dist_buffers(
    buffer_size_ranks: List[Tuple[int, int]],
    local_dist_buffers: Union[Tuple[Tensor], List[Tensor]],
) -> List[Tuple[Tensor, int]]:
  """Split given buffers according to a list of pairs of buffer size and an
    assigned rank.

    Args:
        buffer_size_ranks (List[Tuple[int, int]]): a list of pairs of buffer
        size and an assigned rank.
        local_dist_buffers (Union[Tuple[Tensor], List[Tensor]]): a list of
        tensors to be split

    Returns:
        buffer_ranks (List[Tuple[Tensor, int]]): A list of pairs of a view
        tensor and an assigned rank

    Example:
        tensor0 = tensor(1024)
        tensor1 = tensor(1024)
        buffer_size_ranks = [(128, 0), (64, 0), (512, 1), (256, 0)]
        local_dist_buffers = [tensor0, tensor1]
        -> buffer_ranks = [
             (tensor0's view(  0-128 bytes), 0),
             (tensor0's view(128-192 bytes), 0),
             (tensor1's view(  0-512 bytes), 1),
             (tensor0's view(192-448 bytes), 0),
           ]

    """

  # Create list of lists containing local views of each split tensor for each
  # rank.
  split_tensors_list = []
  for rank, local_dist_buffer in enumerate(local_dist_buffers):
    buffer_sizes = [s for s, r in buffer_size_ranks if r == rank]
    remainder_size = local_dist_buffer.size(0) - sum(buffer_sizes)
    split_tensors = torch.split(local_dist_buffer,
                                buffer_sizes + [remainder_size])
    split_tensors_list.append(split_tensors)

  # Obtain ordered buffer ranks containing (view of local buffer, rank).
  buffer_ranks = []
  buffer_indices = [0] * len(
      local_dist_buffers
  )  # index counter for each rank for obtaining right buffer
  for _, rank in buffer_size_ranks:
    buffer_ranks.append((split_tensors_list[rank][buffer_indices[rank]], rank))
    buffer_indices[rank] += 1

  return buffer_ranks


###### ENUM CLASSES ######
class PreconditionerType(enum.Enum):
  FULL = 0
  DIAGONAL = 1


class GraftingType(enum.Enum):
  NONE = 0
  SGD = 1
  ADAGRAD = 2
  RMSPROP = 3
  ADAM = 4
  ADAGRAD_NORMALIZED = 5
  RMSPROP_NORMALIZED = 6
  ADAM_NORMALIZED = 7
  LARS = 8
  LAMB = 9


class CommunicationDType(enum.Enum):
  DEFAULT = 0
  FP16 = 1
  BF16 = 2
  FP32 = 3


class LargeDimMethod(enum.Enum):
  DIAGONAL = 0
  ADAGRAD = 1
  BLOCKING = 2


# DType mapping for quantized communications.
dtype_mapping = {
    0: "DEFAULT", 1: torch.float16, 2: torch.bfloat16, 3: torch.float32
}


###### MERGING AND BLOCKING HELPER FUNCTIONS ######
def merge_small_dims(tensor_shape: List[int], threshold: int) -> List[int]:
  """Reshapes tensor by merging small dimensions.

    Args:
        tensor_shape (List[int]): The shape of the tensor.
        threshold (int): Threshold on the maximum size of each dimension.

    Returns:
        new_tensor_shape (List[int]): New tensor shape.

    """

  new_tensor_shape = [tensor_shape[0]]
  for next_tensor_shape in tensor_shape[1:]:
    new_dimension = new_tensor_shape[-1] * next_tensor_shape
    if (new_tensor_shape[-1] == 1 or next_tensor_shape == 1 or
        new_dimension <= threshold):
      new_tensor_shape[-1] = new_dimension
    else:
      new_tensor_shape.append(next_tensor_shape)

  return new_tensor_shape


def multi_dim_split(tensor: Tensor, splits: List[int]) -> List[Tensor]:
  """Chunks tensor across multiple dimensions based on splits.

    Args:
        tensor (Tensor): Gradient or tensor to split.
        splits (List[int]): List of sizes for each block or chunk along each
            dimension.

    Returns:
        split_grad (List[Tensor]): List of tensors.

    """
  split_tensors = [tensor]
  for dim, split in enumerate(splits):
    split_tensors = [
        s for t in split_tensors for s in torch.split(t, split, dim=dim)
    ]
  return split_tensors


def multi_dim_cat(split_tensors: List[Tensor], num_splits: List[int]) -> Tensor:
  """Concatenates multiple tensors to form single tensor across multiple
    dimensions.

    Args:
        split_tensor (List[Tensor]): List of tensor splits or blocks.
        num_splits (List[int]): Number of splits/blocks.

    Returns:
        merged_tensor (Tensor): Merged tensor.

    """
  merged_tensor = split_tensors
  for dim, split in reversed(list(enumerate(num_splits))):
    if split > 0:
      merged_tensor = [
          torch.cat(merged_tensor[i:i + split], dim=dim)
          for i in range(0, len(merged_tensor), split)
      ]
  assert len(merged_tensor) == 1
  return merged_tensor[0]


###### PRECONDITIONER CLASSES ######
class Preconditioner(OptimizerModule):
  """Preconditioner base class.

    Args:
        param (Tensor): Parameter of interest.

    """

  def __init__(
      self,
      param,
  ):
    super().__init__()
    self._parameter_count = 0
    self._dims = list(param.size())
    self._block_count = 1
    self._num_bytes = 0

  def update_preconditioners(self, grad: Tensor, iteration: int) -> None:
    pass

  def precondition(self, grad: Tensor,
                   iteration: int) -> Union[Tensor, List[Tensor]]:
    pass

  def compute_norm(self, grad: Tensor, iteration: int) -> Tensor:
    pass

  @property
  def block_count(self) -> int:
    return self._block_count

  @property
  def parameter_count(self) -> int:
    return self._parameter_count


class DistributedPreconditioner(Preconditioner):
  """Distributed Preconditioner class.

    Builds on top of Preconditioner base class to instantiate group and
    distributed buffer information.

    Args:
        param (Tensor): Parameter of interest.
        group (Optional[dist.ProcessGroup]): Process group for distributed
            computation. (Default: None)
        group_source_rank (int): Source rank (or owner) of preconditioner data.
            (Default: 0)
        dist_buffer (Optional[Tensor]): Distributed buffer for distributed
            computation. (Default: None)
        communication_dtype (CommunicationDType): Datatype for communication
            between ranks. (Default: DEFAULT)

    """

  def __init__(
      self,
      param: Tensor,
      group: Optional[dist.ProcessGroup] = None,
      group_source_rank: int = 0,
      dist_buffer: Optional[Tensor] = None,
      communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
  ):
    super().__init__(param)

    # Initialize distributed buffer and source rank.
    # Note that Adagrad dtype is the same as parameter/gradient dtype.
    group_size = (
        dist.get_world_size(group)
        if dist.is_initialized() and group is not None else 1)
    group_rank = (
        dist.get_rank(group)
        if dist.is_initialized() and group is not None else 0)

    # Initializes distributed buffer if dist_buffer is provided;
    # otherwise, sets to default values.
    if dist_buffer is not None:
      requested_dist_buffer_size = self.get_dist_buffer_size(
          param, communication_dtype)

      # We ignore the remainder chunk.
      self._dist_buffer = (
          dist_buffer.split(requested_dist_buffer_size)
          [0].view(param.dtype if communication_dtype == CommunicationDType
                   .DEFAULT else dtype_mapping[communication_dtype.value]).view(
                       param.shape))

    else:
      self._dist_buffer = None

    # Flag for determining whether or not we are on source rank.
    # If group is None and group source rank is not provided, then this will be
    # True.
    self._group_source_rank = group_source_rank
    self._on_source_rank = group_rank == group_source_rank

    # Initialize device mesh and placements for DTensor.
    global_size = dist.get_world_size() if dist.is_initialized() else 1
    self._device_mesh_ranks = [
        *range(group_source_rank % group_size, global_size, group_size)
    ]

  def get_debug_info(self) -> Dict[str, str]:
    debug_info = {
        "name": "DistributedPreconditioner",
        "dims": str(tuple(self._dims)),
        "num_bytes": str(self._num_bytes),
    }
    if self._dist_buffer is not None:
      debug_info.update({
          "group_source_rank": str(self._group_source_rank),
          "dist_buffer": f"({tuple(self._dist_buffer.size())}, "
                         f"dtype={self._dist_buffer.dtype})",
      })

    return debug_info

  def combine_and_split_dims(self, p: Tensor) -> List[Tensor]:
    return [p]

  def get_split_parameters(self, param: Tensor) -> List[Tensor]:
    return self.combine_and_split_dims(param)

  @staticmethod
  def get_dist_buffer_size(param,
                           communication_dtype: CommunicationDType) -> int:
    # Get the buffer size in bytes
    return int(
        math.prod(param.shape) *
        get_dtype_size(param.dtype if communication_dtype == CommunicationDType
                       .DEFAULT else dtype_mapping[communication_dtype.value]))

  @property
  def on_source_rank(self) -> bool:
    return self._on_source_rank

  def preconditioned_grad_to_dist_buffer(self, grad: Tensor,
                                         iteration: int) -> None:
    if self._on_source_rank and self._dist_buffer is not None:
      self._dist_buffer.copy_(self.precondition(grad, iteration))

  def get_from_dist_buffer(self) -> Optional[Tensor]:
    # _dist_buffer must be returned regardless of the owner.
    # When this function is called, this buffer should be already filled
    # by a communication operation.
    return self._dist_buffer

  def get_split_dist_buffers(self) -> List[Tensor]:
    return [self._dist_buffer] if self._dist_buffer is not None else []

  def get_num_bytes(self, group_rank: int = -1) -> int:
    return (self._num_bytes
            if group_rank == -1 or self._group_source_rank == group_rank else 0)


class AdagradPreconditioner(DistributedPreconditioner):
  """Adagrad / Adam / RMSProp preconditioner for a generic layer.

    Stores preconditioner using same format as parameter p. Operations are
    performed in-place.

    NOTE: Does not support sparse gradients at this time.

    To enable Adagrad, set beta2 = 1.0.
    To enable RMSProp, set beta2 = 0.999.
    To enable Adam, set beta2 = 0.999, use_bias_correction = True.

    Other variants can also be specified.

    Args:
        param (Tensor): Parameter of interest.
        beta1 (float): Exponential moving average factor for gradient.
            (Default: 0.0)
        beta2 (float): Exponential moving average factor for Shampoo factor
            matrices. If beta2 = 1., will use unweighted sum. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure
            positive definiteness. (Default: 1e-10)
        use_bias_correction (bool): Flag for using bias correction.
            (Default: False)
        idx (Union[None, str, int]): Layer index (for logging purposes).
            (Default: None)
        group (Optional[dist.ProcessGroup]): Process group for distributed
            computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner.
            (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation.
            (Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly.
            Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication
            between ranks. (Default: DEFAULT)

    """

  def __init__(
      self,
      param: Tensor,
      beta1: float = 0.0,
      beta2: float = 1.0,
      epsilon: float = 1e-10,
      use_bias_correction: bool = True,
      idx: Union[None, str, int] = None,
      group: Optional[dist.ProcessGroup] = None,
      group_source_rank: int = 0,
      dist_buffer: Optional[Tensor] = None,
      use_dtensor: bool = True,
      communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
  ):
    super().__init__(
        param,
        group,
        group_source_rank,
        dist_buffer,
        communication_dtype,
    )
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._preconditioner = allocate_distributed_tensor(
        param.shape,
        dtype=param.dtype,
        device=param.device,
        device_mesh_ranks=self._device_mesh_ranks,
        use_dtensor=use_dtensor,
    )
    self._idx = idx
    self._use_bias_correction = use_bias_correction
    self._bias_correction2 = 1.0
    self._parameter_count += self._preconditioner.numel()
    self._filtered_grad = (
        allocate_distributed_tensor(
            param.shape,
            dtype=param.dtype,
            device=param.device,
            device_mesh_ranks=self._device_mesh_ranks,
            use_dtensor=use_dtensor,
        ) if self._beta1 != 0.0 else None)

    if self._idx is not None:
      self._preconditioner_idx = str(self._idx) + "." + str(0)
      logger.info(
          f"Diagonal Adagrad Preconditioner {self._preconditioner_idx} with "
          f"Parameter {self._idx}")

  def update_preconditioners(self, grad: Tensor, iteration: int) -> None:
    if not self._on_source_rank:
      return

    preconditioner = use_local_tensor(self._preconditioner)

    if self._beta2 == 1.0:
      preconditioner.addcmul_(grad, grad, value=1)
    else:
      preconditioner.mul_(self._beta2).addcmul_(
          grad, grad, value=1 - self._beta2)

    if self._use_bias_correction and self._beta2 < 1.0:
      self._bias_correction2 = 1.0 - self._beta2**iteration

  def precondition(self, grad: Tensor, iteration: int) -> Tensor:
    if not self._on_source_rank:
      return grad

    with profiler.record_function("## adagrad:precondition ##"):
      if self._beta1 != 0.0:
        # Compute bias corrections.
        bias_correction1 = (1.0 - self._beta1**iteration
                            if self._use_bias_correction else 1.0)
        # Compute exponential moving average of the gradient (with potential
        # bias correction).
        filtered_grad = use_local_tensor(self._filtered_grad)
        filtered_grad.mul_(self._beta1).add_(grad, alpha=1 - self._beta1)
        grad.copy_(filtered_grad / bias_correction1)

      denom = ((use_local_tensor(self._preconditioner) /
                self._bias_correction2).sqrt().add_(self._epsilon))
      grad.div_(denom)
      return grad

  def compute_norm(self, grad: Tensor, iteration: int):
    if not self._on_source_rank:
      return torch.as_tensor(1.0)  # return cheap tensor

    denom = ((use_local_tensor(self._preconditioner) /
              self._bias_correction2).sqrt().add_(self._epsilon))
    adagrad_nrm = torch.linalg.norm(grad / denom)
    return adagrad_nrm

  def to(self, device: Union[None, torch.device] = None):
    if device is not None:
      self._preconditioner = self._preconditioner.to(device=device)

  def num_preconditioners(self) -> int:
    return 1


class ShampooKroneckerFactor(OptimizerModule):
  """Shampoo Kronecker Factor Matrix / Preconditioner data class."""

  def __init__(
      self,
      preconditioner_type: PreconditionerType,
      factor_matrix: Tensor,
      inv_factor_matrix: Optional[Tensor] = None,
      index: Optional[str] = None,
      is_diagonal: bool = True,
  ):
    super().__init__()
    self.preconditioner_type = preconditioner_type
    self.factor_matrix = factor_matrix
    self.inv_factor_matrix = inv_factor_matrix
    self.index = index
    self.is_diagonal = is_diagonal


class ShampooPreconditioner(DistributedPreconditioner):
  """Shampoo preconditioners for some generic layer.

    NOTE: Does not support sparse gradients at this time.

    Args:
        param (Tensor): Parameter of interest.
        beta1 (float): Exponential moving average factor for gradient.
            (Default: 0.0)
        beta2 (float): Exponential moving average factor for Shampoo factor
            matrices. If beta2 = 1., will use unweighted sum. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure
            positive definiteness. (Default: 1e-12)
        exponent_override (int, List[int]): inverse root to use in Shampoo. If a
            list [l1, l2, ..., lp], then we will  use -1 / l1 for 1-D tensor
            (vectors), -1 / l2 for 2-D tensors (matrices), and so on. If the
            order of the tensor exceeds the length of the list, we revert to
            using the default value. If 0 is used, uses the default inverse root
            -1 / (2 * o), where o is the order of the tensor. (Default: 0)
        exponent_multiplier (float): number to be multiplied to the numerator of
            the inverse root, i.e., eta where the exponent is -eta / (2 * p).
            (Default: 1.0)
        use_bias_correction (bool): Flag for using bias correction.
            (Default: True)
        diagonal_threshold (int): Threshold for using diagonal preconditioners.
            If None, disabled. (Default: None)
        dtype (torch.dtype): Data type for accumulating and computing root
            inverse of preconditioners. (Default: torch.float)
        idx (Union[None, int, str]): Layer index (for logging purposes).
            (Default: None)
        start_preconditioning_step (int): initial delay before starting to
            compute root inverse. Applies grafting method beforehand.
            (default: 0)
        grafting_type (GraftingType): Selects grafting method.
            (Default: GraftingType.NONE)
        grafting_beta2 (float): Exponential moving average factor for grafting
            method. (Default: 1.0)
        grafting_epsilon (float): Epsilon for grafting method. (Default: 1e-3)
        group (Optional[dist.ProcessGroup]): Process group for distributed
            computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner.
            (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation.
            (Default: None)
        use_protected_eigh (bool): Flag for using two guards to prevent failures
            of torch.linalg.eigh. (Default: True)
            1. Attempts to compute root inverse in preconditioner_dtype
                precision.
            2. Attempts to recompute the eigendecomposition if using
                lower-precision fails.
            3. Otherwise, re-uses previous inverse factor matrix when both root
                inverse computations fail.
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly.
            Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication
            between ranks. (Default: DEFAULT)

    """

  def __init__(
      self,
      param,
      beta1: float = 0.0,
      beta2: float = 1.0,
      epsilon: float = 1e-12,
      exponent_override: Union[int, List[int]] = 0,
      exponent_multiplier: float = 1.0,
      use_bias_correction: bool = True,
      diagonal_threshold: Union[None, int] = None,
      dtype: torch.dtype = torch.float,
      idx: Union[None, int, str] = None,
      start_preconditioning_step: int = 0,
      grafting_type: GraftingType = GraftingType.NONE,
      grafting_beta2: float = 1.0,
      grafting_epsilon: float = 1e-3,
      group: Optional[dist.ProcessGroup] = None,
      group_source_rank: int = 0,
      dist_buffer: Optional[Tensor] = None,
      use_protected_eigh: bool = True,
      use_dtensor: bool = True,
      communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
  ):
    super().__init__(
        param,
        group,
        group_source_rank,
        dist_buffer,
        communication_dtype,
    )

    # Initialize parameters.
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._exponent_override = exponent_override
    self._exponent_multiplier = exponent_multiplier
    self._diagonal_threshold = diagonal_threshold
    self._dtype = dtype
    self._use_bias_correction = use_bias_correction
    self._bias_correction2 = 1.0
    self._order = param.dim()
    self._idx = idx
    self._grafting_type = grafting_type
    self._start_preconditioning_step = start_preconditioning_step
    self._use_protected_eigh = use_protected_eigh
    self._communication_dtype = communication_dtype
    self._filtered_grad = (
        allocate_distributed_tensor(
            param.shape,
            dtype=dtype,
            device=param.device,
            device_mesh_ranks=self._device_mesh_ranks,
            use_dtensor=use_dtensor,
        ) if beta1 != 0.0 else None)

    # Compute root.
    self._root = self._get_root_from_exponent_override(self._exponent_override,
                                                       self._order)

    # Initialize lists for preconditioners, inverse preconditioners, types, and
    # ranks.
    self._preconditioners = []

    for k, dim in enumerate(self._dims):
      index = str(self._idx) + "." + str(k) if self._idx else None

      # Creates a diagonal Shampoo preconditioner if dimension is larger than
      # self._diagonal_threshold.
      if (self._diagonal_threshold is not None and
          dim > self._diagonal_threshold):
        preconditioner_type = PreconditionerType.DIAGONAL
        factor_matrix = allocate_distributed_tensor(
            dim,
            dtype=param.dtype,
            device=param.device,
            device_mesh_ranks=self._device_mesh_ranks,
            use_dtensor=use_dtensor,
        )
        inv_factor_matrix = None
        num_params = dim

      # Otherwise, generates a full Shampoo preconditioner.
      else:
        preconditioner_type = PreconditionerType.FULL
        factor_matrix = allocate_distributed_tensor(
            (dim, dim),
            dtype=self._dtype,
            device=param.device,
            device_mesh_ranks=self._device_mesh_ranks,
            use_dtensor=use_dtensor,
        )
        inv_factor_matrix = allocate_distributed_tensor(
            (dim, dim),
            dtype=self._dtype,
            device=param.device,
            device_mesh_ranks=self._device_mesh_ranks,
            use_dtensor=use_dtensor,
        )
        num_params = 2 * dim**2

      # Counts parameters and adds to lists.
      self._parameter_count += num_params
      self._num_bytes += num_params * get_dtype_size(dtype)
      self._preconditioners.append(
          ShampooKroneckerFactor(
              preconditioner_type,
              factor_matrix,
              inv_factor_matrix,
              index,
          ))

    # Initialize grafting method.
    if self._grafting_type == GraftingType.NONE:
      self._grafting = None
    elif self._grafting_type == GraftingType.SGD:
      self._grafting = SGDGrafting(param)
    elif self._grafting_type == GraftingType.ADAGRAD:
      self._grafting = AdagradGrafting(
          param,
          epsilon=grafting_epsilon,
          group=group,
          group_source_rank=group_source_rank,
          dist_buffer=dist_buffer,
          use_dtensor=use_dtensor,
          communication_dtype=communication_dtype,
      )
    elif self._grafting_type == GraftingType.RMSPROP:
      self._grafting = RMSPropGrafting(
          param,
          beta2=grafting_beta2,
          epsilon=grafting_epsilon,
          group=group,
          group_source_rank=group_source_rank,
          dist_buffer=dist_buffer,
          use_dtensor=use_dtensor,
          communication_dtype=communication_dtype,
      )
    elif self._grafting_type == GraftingType.ADAM:
      self._grafting = AdamGrafting(
          param,
          beta2=grafting_beta2,
          epsilon=grafting_epsilon,
          group=group,
          group_source_rank=group_source_rank,
          dist_buffer=dist_buffer,
          use_dtensor=use_dtensor,
          communication_dtype=communication_dtype,
      )
    elif self._grafting_type == GraftingType.ADAGRAD_NORMALIZED:
      self._grafting = AdagradNormalizedGrafting(
          param,
          epsilon=grafting_epsilon,
          group=group,
          group_source_rank=group_source_rank,
          dist_buffer=dist_buffer,
          use_dtensor=use_dtensor,
          communication_dtype=communication_dtype,
      )
    elif self._grafting_type == GraftingType.RMSPROP_NORMALIZED:
      self._grafting = RMSPropNormalizedGrafting(
          param,
          beta2=grafting_beta2,
          epsilon=grafting_epsilon,
          group=group,
          group_source_rank=group_source_rank,
          dist_buffer=dist_buffer,
          use_dtensor=use_dtensor,
          communication_dtype=communication_dtype,
      )
    elif self._grafting_type == GraftingType.ADAM_NORMALIZED:
      self._grafting = AdamNormalizedGrafting(
          param,
          beta2=grafting_beta2,
          epsilon=grafting_epsilon,
          group=group,
          group_source_rank=group_source_rank,
          dist_buffer=dist_buffer,
          use_dtensor=use_dtensor,
          communication_dtype=communication_dtype,
      )
    elif self._grafting_type == GraftingType.LARS:
      self._grafting = SGDGrafting(param)
    elif self._grafting_type == GraftingType.LAMB:
      self._grafting = AdamGrafting(
          param,
          beta2=grafting_beta2,
          epsilon=grafting_epsilon,
          group=group,
          group_source_rank=group_source_rank,
          dist_buffer=dist_buffer,
          use_dtensor=use_dtensor,
      )
    else:
      raise ValueError(f"Invalid Grafting Type {self._grafting_type}!")

    # Counts parameters for grafted method.
    self._parameter_count += getattr(self._grafting, "parameter_count", 0)
    self._num_bytes += getattr(self._grafting, "num_bytes", 0)

  def get_debug_info(self) -> Dict[str, str]:
    debug_info = super().get_debug_info()
    preconditioner_strs = []
    for preconditioner in self._preconditioners:
      matrix_sizes = [tuple(preconditioner.factor_matrix.size())
                     ] + ([tuple(preconditioner.inv_factor_matrix.size())] if
                          preconditioner.inv_factor_matrix is not None else [])
      preconditioner_strs.append(
          f"{'Diagonal' if preconditioner.is_diagonal else 'Full'}"
          f"({preconditioner.index}, "
          f"[{', '.join(str(s) for s in matrix_sizes)}], {self._dtype})")

    debug_info.update({
        "name": "ShampooPreconditioner",
        "idx": str(self._idx),
        "preconditioners": f"[{', '.join(preconditioner_strs)}]",
    })
    return debug_info

  @staticmethod
  def _get_root_from_exponent_override(exponent_override: Union[int, List[int]],
                                       order: int) -> int:
    """Retrieves the appropriate root from the exponent override parameter.

        Args:
            exponent_override (int, List[int]): Exponent override int or list.
            order (List[int]): Order of the tensor of interest.

        Returns:
            root (int): Root to use in Shampoo.

        """
    if isinstance(exponent_override, list):
      if order > len(exponent_override):
        return 2 * order
      else:
        return exponent_override[order - 1]
    else:
      return 2 * order if exponent_override == 0 else exponent_override

  def update_preconditioners(self, grad: Tensor, iteration: int) -> None:
    if not self._on_source_rank:
      return

    with profiler.record_function("## shampoo:update_preconditioners ##"):
      for k, (dim, preconditioner) in enumerate(
          zip(self._dims, self._preconditioners)):
        factor_matrix = use_local_tensor(preconditioner.factor_matrix)

        if self._beta2 != 1.0:
          factor_matrix.mul_(self._beta2)

        # Update diagonal Shampoo preconditioner.
        if preconditioner.preconditioner_type == PreconditionerType.DIAGONAL:
          diagonal_or_outer_product = torch.linalg.norm(
              grad.transpose(0, k).contiguous().view(dim, -1),
              dim=1,
          ).pow(2)

        # Update full Shampoo preconditioner.
        else:
          contract_idx = [*range(k)] + [*range(k + 1, self._order)]
          diagonal_or_outer_product = torch.tensordot(
              grad,
              grad,
              dims=(contract_idx, contract_idx),
          )
          if diagonal_or_outer_product.dtype != self._dtype:
            diagonal_or_outer_product = diagonal_or_outer_product.to(
                dtype=self._dtype)

        factor_matrix.add_(
            diagonal_or_outer_product,
            alpha=1 - self._beta2 if self._beta2 != 1.0 else 1.0,
        )

      # Update grafting preconditioner.
      if self._grafting_type != GraftingType.NONE:
        self._grafting.update_preconditioners(grad, iteration)

      if self._use_bias_correction and self._beta2 < 1.0:
        self._bias_correction2 = 1.0 - self._beta2**iteration

  def _shampoo_precondition(self, grad: Tensor, iteration: int) -> Tensor:
    if not self._on_source_rank:
      return grad  # An invalid tensor that can be returned at the lowest cost.

    preconditioned_grad = grad.clone()
    for k, preconditioner in enumerate(self._preconditioners):
      # Use local versions of factor and inv factor matrices.
      factor_matrix = use_local_tensor(preconditioner.factor_matrix)
      inv_factor_matrix = use_local_tensor(preconditioner.inv_factor_matrix)

      # To handle diagonal case, requires not transposing the tensor.
      if self._diagonal_threshold is not None:

        # Precondition using diagonal preconditioner.
        if preconditioner.preconditioner_type == PreconditionerType.DIAGONAL:
          denom = (factor_matrix / self._bias_correction2).add_(self._epsilon)
          preconditioned_grad.div_(
              denom.pow(-self._exponent_multiplier /
                        self._root)[(None,) * k + (...,) + (None,) *
                                    (self._order - k - 1)])

        # Precondition using full Shampoo preconditioner.
        # Uses einsum in order to avoid transposing.
        else:
          gradient_idx = [*range(1, self._order + 1)]
          matrix_product_idx = deepcopy(gradient_idx)
          matrix_product_idx[k] = 0
          preconditioned_grad = torch.einsum(
              inv_factor_matrix,
              [0, k + 1],
              preconditioned_grad,
              gradient_idx,
              matrix_product_idx,
          )

      # Handles full Shampoo preconditioner case more efficiently but
      # transposes the tensor continually.
      else:
        preconditioned_grad = torch.tensordot(preconditioned_grad,
                                              inv_factor_matrix, [[0], [0]])

      if preconditioned_grad.dtype != grad.dtype:
        preconditioned_grad.to(dtype=grad.dtype)

    # Apply grafting.
    if self._grafting_type != GraftingType.NONE:
      grafting_norm = self._grafting.direction_norm(grad, iteration)
      shampoo_norm = torch.linalg.norm(preconditioned_grad)
      preconditioned_grad.mul_(grafting_norm).div_(shampoo_norm + 1e-16)

    return preconditioned_grad

  def _graft_precondition(self, grad: Tensor, iteration: int) -> Tensor:
    if not self._on_source_rank:
      return grad  # An invalid tensor that can be returned at the lowest cost.
    return (self._grafting.precondition(grad, iteration)
            if self._grafting_type != GraftingType.NONE else grad)

  def precondition(self, grad: Tensor, iteration: int) -> Tensor:
    if not self._on_source_rank:
      return grad  # An invalid tensor that can be returned at the lowest cost.

    use_graft_precondition = iteration < self._start_preconditioning_step
    if self._beta1 != 0.0:
      # Compute bias corrections.
      bias_correction1 = (1.0 - self._beta1**iteration
                          if self._use_bias_correction else 1.0)
      # Compute exponential moving average of the gradient (with potential bias
      # correction).
      filtered_grad = use_local_tensor(self._filtered_grad)
      filtered_grad.mul_(self._beta1).add_(grad, alpha=1 - self._beta1)
      grad.copy_(filtered_grad / bias_correction1)

    with profiler.record_function(
        "## shampoo:graft_precondition ##"
        if use_graft_precondition else "## shampoo:shampoo_precondition ##"):
      return (self._graft_precondition if use_graft_precondition else
              self._shampoo_precondition)(grad, iteration)

  def compute_root_inverse(self) -> None:
    if not self._on_source_rank:
      return

    for k, preconditioner in enumerate(self._preconditioners):
      # Use local versions of factor and inv factor matrices.
      factor_matrix = use_local_tensor(preconditioner.factor_matrix)
      inv_factor_matrix = use_local_tensor(preconditioner.inv_factor_matrix)

      # Check that this is a full Shampoo preconditioner.
      if preconditioner.preconditioner_type == PreconditionerType.FULL:

        # For tracking diagonality of the preconditioner.
        # Checks if the preconditioner is currently diagonal, then checks
        # whether or not the update matrix is diagonal.
        if preconditioner.is_diagonal and not check_diagonal(factor_matrix):
          preconditioner.is_diagonal = False
          logger.debug(
              f"Preconditioner {preconditioner.index} is not diagonal.")

        # Add epsilon term and incorporate bias correction.
        bias_corrected_preconditioner = factor_matrix / self._bias_correction2

        # Check for nan or inf values.
        if torch.any(torch.isnan(bias_corrected_preconditioner)):
          logger.warning(
              f"Encountered nan values in preconditioner {self._idx}.{k}!")
        elif torch.any(torch.isinf(bias_corrected_preconditioner)):
          logger.warning(
              f"Encountered inf values in preconditioner {self._idx}.{k}!")

        # Compute inverse preconditioner.
        # If reuse_previous_inv_factor_matrix is True, will reuse previous
        # matrix if matrix inverse root computation fails.
        try:
          computed_inv_factor_matrix = matrix_inverse_root(
              A=bias_corrected_preconditioner,
              root=self._root,
              epsilon=self._epsilon,
              exponent_multiplier=self._exponent_multiplier,
              is_diagonal=preconditioner.is_diagonal,
              retry_double_precision=self._use_protected_eigh,
          ).to(dtype=self._dtype)

          # check if we encounter NaN or inf values in computed inverse matrix.
          if torch.any(torch.isnan(computed_inv_factor_matrix)):
            raise ValueError(
                "Encountered nan values in root inv preconditioner "
                f"{self._idx}.{k}!")
          if torch.any(torch.isinf(computed_inv_factor_matrix)):
            raise ValueError(
                "Encountered inf values in root inv preconditioner "
                f"{self._idx}.{k}!")

          inv_factor_matrix.copy_(computed_inv_factor_matrix)

        except Exception as exception:  # pylint: disable=broad-exception-caught
          if (not self._use_protected_eigh or
              "values in root inv preconditioner" in str(exception)):
            raise exception
          logger.warning(
              "Matrix inverse root computation failed for preconditioner "
              f"{self._idx}.{k} with exception {exception}. Using previous "
              "inv_factor_matrix and continuing...")

  def compute_root_inverse_residuals(
      self,) -> Tuple[List[Tensor], List[Tensor]]:
    relative_errors = []
    relative_residuals = []

    if not self._on_source_rank:
      return relative_errors, relative_residuals

    for preconditioner in self._preconditioners:
      if preconditioner.preconditioner_type == PreconditionerType.FULL:
        # Use local versions of factor and inv factor matrices.
        factor_matrix = use_local_tensor(preconditioner.factor_matrix)
        inv_factor_matrix = use_local_tensor(preconditioner.inv_factor_matrix)

        bias_corrected_preconditioner = factor_matrix / self._bias_correction2
        (
            relative_error,
            relative_residual,
        ) = compute_matrix_root_inverse_residuals(
            bias_corrected_preconditioner,
            inv_factor_matrix,
            self._root,
            self._epsilon,
            self._exponent_multiplier,
        )
        relative_errors.append(relative_error)
        relative_residuals.append(relative_residual)

    return (
        relative_errors,
        relative_residuals,
    )

  def compute_norm(self, grad: Tensor, iteration: int) -> Tensor:
    return torch.linalg.norm(self.precondition(grad, iteration))

  def to(self, device: Union[None, torch.device] = None):
    if device is not None:
      for preconditioner in self._preconditioners:
        preconditioner.to(device)
      if self._grafting is not None:
        self._grafting.to(device=device)

  def num_preconditioners(self) -> int:
    return 1

  def reset_preconditioners(self) -> None:
    for preconditioner in self._preconditioners:
      preconditioner.factor_matrix.zero_()


class BlockShampooPreconditioner(DistributedPreconditioner):
  """Shampoo with blocking applied to the parameters.

    NOTE: Does not support sparse gradients at this time.

    Args:
        param (Tensor): Parameter of interest.
        beta1 (float): Exponential moving average factor for gradient.
          (Default: 0.0)
        beta2 (float): Exponential moving average factor for Shampoo factor
          matrices. If beta2 = 1., will use unweighted sum. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure
          positive definiteness. (Default: 1e-12)
        exponent_override (int, List[int]): inverse root to use in Shampoo. If a
          list [l1, l2, ..., lp], then we will use -1 / l1 for 1-D tensor
          (vectors), -1 / l2 for 2-D tensors (matrices), and so on. If the order
          of the tensor exceeds the length of the list, we revert to using the
          default value. If 0 is used, uses the default inverse root -1 /
          (2 * o), where o is the order of the tensor. (Default: 0)
        exponent_multiplier (float): number to be multiplied to the numerator of
          the inverse root, i.e., eta where the exponent is -eta / (2 * p).
          (Default: 1.0)
        use_bias_correction (bool): Flag for using bias correction.
          (Default: True)
        block_size (int): Block size for blocking large tensors.
          (Default: 1024)
        dtype (torch.dtype): Data type for accumulating and computing root
          inverse of preconditioners. (Default: torch.float)
        idx (Union[None, int, str]): Layer index (for logging purposes).
          (Default: None)
        use_merge_dims (bool): Denotes whether or not dimensions are merged.
          (Default: True)
        cache_split_params (bool): cache split parameters across iterations.
          (Default: False)
        start_preconditioning_step (int): initial delay before starting to
          compute root inverse. Applies grafting method beforehand. (Default: 0)
        grafting_type (LayerwiseGraftingType): Selects grafting method.
          (Default: GraftingType.NONE)
        grafting_beta2 (float): Exponential moving average factor for grafting
          method. (Default: 1.0)
        grafting_epsilon (float): Epsilon for grafting method. (Default: 1e-3)
        group (Optional[dist.ProcessGroup]): Process group for distributed
          computation. (Default: None)
        dist_buffer_ranks (Optional[List[Tuple[Tensor, int]]]): List of
          distributed buffers and their group rank assignments. (Default: None)
        dist_buffer_index (int): Index for getting dist_buffer and rank from
          dist_buffer and rank list. (Default: 0)
        use_protected_eigh (bool): Flag for using two guards to prevent failures
          of torch.linalg.eigh. (Default: True)
            1. Attempts to compute root inverse in preconditioner_dtype
              precision.
            2. Attempts to recompute the eigendecomposition if using
              lower-precision fails.
            3. Otherwise, re-uses previous inverse factor matrix when both root
              inverse computations fail.
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly.
          Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication
          between ranks. (Default: DEFAULT)

    """

  def __init__(
      self,
      param,
      beta1: float = 0.0,
      beta2: float = 1.0,
      epsilon: float = 1e-12,
      exponent_override: Union[int, List[int]] = 0,
      exponent_multiplier: float = 1.0,
      use_bias_correction: bool = True,
      block_size: int = 1024,
      dtype: torch.dtype = torch.float,
      idx: Union[None, int, str] = None,
      use_merge_dims: bool = True,
      cache_split_params: bool = False,
      start_preconditioning_step: int = 0,
      grafting_type: GraftingType = GraftingType.NONE,
      grafting_beta2: float = 1.0,
      grafting_epsilon: float = 1e-3,
      group: Optional[dist.ProcessGroup] = None,
      dist_buffer_ranks: Optional[List[Tuple[Tensor, int]]] = None,
      dist_buffer_index: int = 0,
      use_protected_eigh: bool = True,
      use_dtensor: bool = True,
      communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
  ):
    super().__init__(param,)

    # Set parameters.
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._exponent_override = exponent_override
    self._exponent_multiplier = exponent_multiplier
    self._use_bias_correction = use_bias_correction
    self._block_size = block_size
    self._dtype = dtype
    self._idx = idx
    self._start_preconditioning_step = start_preconditioning_step
    self._use_merge_dims = use_merge_dims
    self._cache_split_params = cache_split_params
    self._original_dims = list(param.size())
    self._merged_dims = (
        merge_small_dims(self._original_dims, self._block_size)
        if self._block_size is not None and use_merge_dims else
        self._original_dims)

    # Construct splits for blocking
    self._splits = [block_size] * len(self._merged_dims)
    self._num_splits = [
        math.ceil(dim / block_size) for dim in self._merged_dims
    ]

    # Construct multiple preconditioners for each block
    self._split_preconditioners = []
    self._split_sizes = []
    self._cached_split_params = []

    split_param = self.combine_and_split_dims(param)
    for i, p in enumerate(split_param):
      self._split_sizes.append(torch.as_tensor(p.shape))
      split_idx = str(idx) + "." + str(i)
      dist_buffer, group_source_rank = (
          dist_buffer_ranks[dist_buffer_index + i]
          if dist_buffer_ranks is not None
          else (None, 0)
      )
      preconditioner = ShampooPreconditioner(
          p,
          beta1=beta1,
          beta2=beta2,
          epsilon=epsilon,
          exponent_override=exponent_override,
          exponent_multiplier=exponent_multiplier,
          use_bias_correction=use_bias_correction,
          dtype=dtype,
          idx=split_idx,
          start_preconditioning_step=start_preconditioning_step,
          grafting_type=grafting_type,
          grafting_beta2=grafting_beta2,
          grafting_epsilon=grafting_epsilon,
          group=group,
          group_source_rank=group_source_rank,
          dist_buffer=dist_buffer,
          use_protected_eigh=use_protected_eigh,
          use_dtensor=use_dtensor,
          communication_dtype=communication_dtype,
      )
      self._split_preconditioners.append(preconditioner)
      self._parameter_count += preconditioner.parameter_count

    self._block_count = len(self._split_preconditioners)
    # Initialize source rank based on whether or not any preconditioner is
    # on source rank.
    self._on_source_rank = any(
        preconditioner.on_source_rank
        for preconditioner in self._split_preconditioners)

  def get_debug_info(self) -> Dict[str, str]:
    debug_info = super().get_debug_info()
    split_preconditioner_strs = []
    for split_preconditioner in self._split_preconditioners:
      info = split_preconditioner.get_debug_info()
      info_str = ', '.join(
          [f'{key}={val}' for key, val in info.items() if key != 'name'])
      split_preconditioner_strs.append(f"{info['name']}({info_str})")

    debug_info.update({
        "name":
            "BlockShampooPreconditioner",
        "idx":
            str(self._idx),
        "preconditioners":
            "[\n  " + (",\n  ".join(split_preconditioner_strs)) + "\n]",
    })
    return debug_info

  def combine_and_split_dims(self, p: Tensor) -> List[Tensor]:
    if self._use_merge_dims:
      p = p.view(self._merged_dims)
    return multi_dim_split(p, self._splits)

  def _multi_dim_cat(self, split_grads: List[Tensor]) -> Tensor:
    preconditioned_grad = multi_dim_cat(split_grads, self._num_splits)
    return (preconditioned_grad.view(self._original_dims)
            if self._use_merge_dims else preconditioned_grad)

  def update_preconditioners(self, grad: Tensor, iteration: int):
    split_grad = self.combine_and_split_dims(grad)
    assert (
        len(split_grad) == self.num_preconditioners()
    ), (f"BlockShampooPreconditioner {self._idx} has "
        f"{self.num_preconditioners()} preconditioners but grad was split into "
        f"{len(split_grad)} blocks!")
    for block_preconditioner, block_grad in zip(
        self._split_preconditioners, split_grad
    ):
      block_preconditioner.update_preconditioners(block_grad, iteration)

  def precondition(self, grad: Tensor, iteration: int) -> Tensor:
    split_grad = self.combine_and_split_dims(grad)
    assert self.num_preconditioners() == len(
        split_grad
    ), (f"BlockShampooPreconditioner {self._idx} has "
      f"{self.num_preconditioners()} preconditioners but grad was split into "
      f"{len(split_grad)} blocks!")
    split_preconditioned_grad = [
        p.precondition(g, iteration) for p,
        g in zip(self._split_preconditioners, split_grad)
    ]
    preconditioned_grad = multi_dim_cat(split_preconditioned_grad,
                                        self._num_splits)
    return (preconditioned_grad.view(self._original_dims)
            if self._use_merge_dims else preconditioned_grad)

  def compute_root_inverse(self) -> None:
    for preconditioner in self._split_preconditioners:
      preconditioner.compute_root_inverse()

  def compute_root_inverse_residuals(
      self,) -> Tuple[List[Tensor], List[Tensor]]:
    relative_errors = []
    relative_residuals = []

    for preconditioner in self._split_preconditioners:
      (
          relative_errors_temp,
          relative_residuals_temp,
      ) = preconditioner.compute_root_inverse_residuals()

      relative_errors += relative_errors_temp
      relative_residuals += relative_residuals_temp

    return (
        relative_errors,
        relative_residuals,
    )

  def compute_norm(self, grad: Tensor, iteration: int) -> Tensor:
    return torch.linalg.norm(self.precondition(grad, iteration))

  @staticmethod
  def get_dist_buffer_sizes(
      param,
      block_size: int,
      use_merge_dims: bool,
      communication_dtype: CommunicationDType,
  ) -> List[int]:
    original_dims = list(param.size())
    merged_dims = (
        merge_small_dims(original_dims, block_size)
        if block_size is not None and use_merge_dims else original_dims)
    splits = [block_size] * len(merged_dims)
    if use_merge_dims:
      param = param.view(merged_dims)
    return [
        ShampooPreconditioner.get_dist_buffer_size(split_param,
                                                   communication_dtype)
        for split_param in multi_dim_split(param, splits)
    ]

  def preconditioned_grad_to_dist_buffer(self, grad: Tensor,
                                         iteration: int) -> None:
    split_grads = self.combine_and_split_dims(grad)
    assert self.num_preconditioners() == len(
        split_grads
    ), (f"BlockShampooPreconditioner {self._idx} has "
        f"{self.num_preconditioners()} preconditioners but grad was split into "
        f"{len(split_grads)} blocks!")
    for preconditioner, grad in zip(self._split_preconditioners, split_grads):
      preconditioner.preconditioned_grad_to_dist_buffer(grad, iteration)

  def get_from_dist_buffer(self) -> Tensor:
    split_grads = [
        preconditioner._dist_buffer
        for preconditioner in self._split_preconditioners
    ]
    return self._multi_dim_cat(split_grads)

  def get_split_dist_buffers(self) -> List[Tensor]:
    return [
        preconditioner._dist_buffer
        for preconditioner in self._split_preconditioners
    ]

  def num_preconditioners(self) -> int:
    return len(self._split_preconditioners)

  def get_split_parameters(self, param: Tensor) -> List[Tensor]:
    if self._cache_split_params:
      if (len(self._cached_split_params) == 0 or
          self._cached_split_params[0].storage().data_ptr() !=
          param.storage().data_ptr()):
        # Cache new split parameters.
        self._cached_split_params = self.combine_and_split_dims(param)
      return self._cached_split_params
    else:
      return self.combine_and_split_dims(param)

  def reset_preconditioners(self) -> None:
    for preconditioner in self._split_preconditioners:
      preconditioner.reset_preconditioners()

  def get_num_bytes(self, group_rank: int = -1) -> int:
    return sum(
        preconditioner.get_num_bytes(group_rank=group_rank)
        for preconditioner in self._split_preconditioners)


###### GRAFTING CLASSES ######
class Grafting(OptimizerModule):
  """Grafting base class.

    We graft the method by storing and maintaining the preconditioner for the
    grafted method. Therefore, any additional modifications including gradient
    EMA/filtering and momentum are not included in grafting.

    Args:
        param (Tensor): Parameter of interest.

    """

  def __init__(self):
    super().__init__()
    self._parameter_count = 0
    self._num_bytes = 0

  def update_preconditioners(self, grad: Tensor, iteration: int):
    pass

  def precondition(self, grad: Tensor, iteration: int) -> Tensor:
    pass

  def direction_norm(self, grad: Tensor, iteration: int) -> Tensor:
    pass

  @property
  def parameter_count(self):
    return self._parameter_count

  @property
  def num_bytes(self) -> int:
    return self._num_bytes


class SGDGrafting(Grafting):
  """SGD grafting.

    Grafts the stochastic gradient method by returning the norm of the gradient.

    Args:
        param (Tensor): Parameter of interest.


    """

  def precondition(self, grad: Tensor, iteration: int) -> Tensor:
    return grad

  def direction_norm(self, grad: Tensor, iteration: int) -> Tensor:
    return torch.linalg.norm(grad)


class AdagradGrafting(Grafting):
  """Adagrad grafting.

    Supports RMSProp and Adam by determining beta2 and use_bias_correction.

    Note: beta1 is not included since that is shared between both Shampoo and
      the grafted optimizer.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will
          use Adagrad update. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure
          positive definiteness. (Default: 1e-10)
        use_bias_correction (bool): Flag for using bias correction.
          (Default: False)
        normalize_gradient (bool): Flag for normalizing the gradient.
          (Default: False)
        group (Optional[dist.ProcessGroup]): Process group for distributed
          computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner.
          (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation.
          (Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly.
          Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication
          between ranks. (Default: DEFAULT)

    """

  def __init__(
      self,
      param: Tensor,
      beta2: float = 1.0,
      epsilon: float = 1e-10,
      use_bias_correction: bool = True,
      normalize_gradient: bool = False,
      group: Optional[dist.ProcessGroup] = None,
      group_source_rank: int = 0,
      dist_buffer: Optional[Tensor] = None,
      use_dtensor: bool = True,
      communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
  ):
    super().__init__()
    self._preconditioner = AdagradPreconditioner(
        param,
        beta2=beta2,
        epsilon=epsilon,
        use_bias_correction=use_bias_correction,
        group=group,
        group_source_rank=group_source_rank,
        dist_buffer=dist_buffer,
        use_dtensor=use_dtensor,
        communication_dtype=communication_dtype,
    )
    self.normalize_gradient = normalize_gradient
    self._parameter_count += self._preconditioner.parameter_count
    self._num_bytes += self._preconditioner.get_num_bytes()

  def _normalize_grad(self, grad: Tensor) -> Tensor:
    return grad / torch.norm(grad) if self.normalize_gradient else grad

  def update_preconditioners(self, grad: Tensor, iteration: int):
    self._preconditioner.update_preconditioners(
        self._normalize_grad(grad), iteration)

  def precondition(self, grad: Tensor, iteration: int) -> Tensor:
    return self._preconditioner.precondition(grad, iteration)

  def direction_norm(self, grad: Tensor, iteration: int) -> Tensor:
    return self._preconditioner.compute_norm(grad, iteration)


class RMSPropGrafting(AdagradGrafting):
  """RMSProp grafting.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. (Default: 0.99)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure
          positive definiteness. (Default: 1e-8)
        group (Optional[dist.ProcessGroup]): Process group for distributed
          computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner.
          (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation.
          (Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly.
          Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication
          between ranks. (Default: DEFAULT)

    """

  def __init__(
      self,
      param,
      beta2: float = 0.99,
      epsilon: float = 1e-8,
      group: Optional[dist.ProcessGroup] = None,
      group_source_rank: int = 0,
      dist_buffer: Optional[Tensor] = None,
      use_dtensor: bool = True,
      communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
  ):
    super().__init__(
        param=param,
        beta2=beta2,
        epsilon=epsilon,
        use_bias_correction=False,
        normalize_gradient=False,
        group=group,
        group_source_rank=group_source_rank,
        dist_buffer=dist_buffer,
        use_dtensor=use_dtensor,
        communication_dtype=communication_dtype,
    )


class AdamGrafting(AdagradGrafting):
  """Adam grafting.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will
          use Adagrad update. (Default: 0.999)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure
          positive definiteness. (Default: 1e-8)
        group (Optional[dist.ProcessGroup]): Process group for distributed
          computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner.
          (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation. 
          Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly.
          Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication
          between ranks. (Default: DEFAULT)

    """

  def __init__(
      self,
      param,
      beta2: float = 0.999,
      epsilon: float = 1e-8,
      group: Optional[dist.ProcessGroup] = None,
      group_source_rank: int = 0,
      dist_buffer: Optional[Tensor] = None,
      use_dtensor: bool = True,
      communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
  ):
    super().__init__(
        param=param,
        beta2=beta2,
        epsilon=epsilon,
        use_bias_correction=True,
        normalize_gradient=False,
        group=group,
        group_source_rank=group_source_rank,
        dist_buffer=dist_buffer,
        use_dtensor=use_dtensor,
        communication_dtype=communication_dtype,
    )


class AdagradNormalizedGrafting(AdagradGrafting):
  """Adagrad grafting with per-parameter normalized gradients.

    Args:
        param (Tensor): Parameter of interest.
        epsilon (float): Epsilon term for regularizing preconditioner to ensure
          positive definiteness. (Default: 1e-10)
        group (Optional[dist.ProcessGroup]): Process group for distributed
          computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner.
          (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation.
          (Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly.
          Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication
          between ranks. (Default: DEFAULT)

    """

  def __init__(
      self,
      param,
      epsilon: float = 1e-10,
      group: Optional[dist.ProcessGroup] = None,
      group_source_rank: int = 0,
      dist_buffer: Optional[Tensor] = None,
      use_dtensor: bool = True,
      communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
  ):
    super().__init__(
        param=param,
        beta2=1.0,
        epsilon=epsilon,
        use_bias_correction=False,
        normalize_gradient=True,
        group=group,
        group_source_rank=group_source_rank,
        dist_buffer=dist_buffer,
        use_dtensor=use_dtensor,
        communication_dtype=communication_dtype,
    )


class RMSPropNormalizedGrafting(AdagradGrafting):
  """RMSProp grafting with per-parameter normalized gradients.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. (Default: 0.99)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure
          positive definiteness. (Default: 1e-8)
        group (Optional[dist.ProcessGroup]): Process group for distributed
          computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner.
          (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation.
          (Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly.
          Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication
          between ranks. (Default: DEFAULT)

    """

  def __init__(
      self,
      param,
      beta2: float = 0.99,
      epsilon: float = 1e-8,
      group: Optional[dist.ProcessGroup] = None,
      group_source_rank: int = 0,
      dist_buffer: Optional[Tensor] = None,
      use_dtensor: bool = True,
      communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
  ):
    super().__init__(
        param=param,
        beta2=beta2,
        epsilon=epsilon,
        use_bias_correction=False,
        normalize_gradient=True,
        group=group,
        group_source_rank=group_source_rank,
        dist_buffer=dist_buffer,
        use_dtensor=use_dtensor,
        communication_dtype=communication_dtype,
    )


class AdamNormalizedGrafting(AdagradGrafting):
  """Adam grafting with per-parameter normalized gradients.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will
          use Adagrad update. (Default: 0.999)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure
          positive definiteness. (Default: 1e-8)
        group (Optional[dist.ProcessGroup]): Process group for distributed
          computation. (Default: None)
        group_source_rank (int): Group rank assigned to preconditioner.
          (Default: 0)
        dist_buffer (Optional[Tensor]): Buffer for distributed computation.
          (Default: None)
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly.
          Otherwise, uses Tensor. (Default: True)
        communication_dtype (CommunicationDType): Datatype for communication
          between ranks. (Default: DEFAULT)

    """

  def __init__(
      self,
      param,
      beta2: float = 0.999,
      epsilon: float = 1e-8,
      group: Optional[dist.ProcessGroup] = None,
      group_source_rank: int = 0,
      dist_buffer: Optional[Tensor] = None,
      use_dtensor: bool = True,
      communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
  ):
    super().__init__(
        param=param,
        beta2=beta2,
        epsilon=epsilon,
        use_bias_correction=True,
        normalize_gradient=True,
        group=group,
        group_source_rank=group_source_rank,
        dist_buffer=dist_buffer,
        use_dtensor=use_dtensor,
        communication_dtype=communication_dtype,
    )


class NewtonConvergenceFlag(enum.Enum):
  REACHED_MAX_ITERS = 0
  CONVERGED = 1


class RootInvMethod(enum.Enum):
  EIGEN = 0
  NEWTON = 1


def check_diagonal(A: Tensor) -> Tensor:
  """Checks if symmetric matrix is diagonal."""

  A_shape = A.shape
  if len(A_shape) != 2:
    raise ValueError("Matrix is not 2-dimensional!")

  m, n = A_shape
  if m != n:
    raise ValueError("Matrix is not square!")

  return ~torch.any(A.reshape(-1)[:-1].reshape(m - 1, n + 1)[:, 1:].bool())


def matrix_inverse_root(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    exponent_multiplier: float = 1.0,
    root_inv_method: RootInvMethod = RootInvMethod.EIGEN,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    is_diagonal: Union[Tensor, bool] = False,
    retry_double_precision: bool = True,
) -> Tensor:
  """Computes matrix root inverse of square symmetric positive definite matrix.

    Args:
        A (Tensor): Square matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root.
          (Default: 0.0)
        exponent_multiplier (float): exponent multiplier in the eigen method
          (Default: 1.0)
        root_inv_method (RootInvMethod): Specifies method to use to compute root
          inverse. (Default: RootInvMethod.EIGEN)
        max_iterations (int): Maximum number of iterations for coupled Newton
          iteration. (Default: 1000)
        tolerance (float): Tolerance for computing root inverse using coupled
          Newton iteration. (Default: 1e-6)
        is_diagonal (Tensor, bool): Flag for whether or not matrix is diagonal.
          If so, will compute root inverse by computing
            root inverse of diagonal entries. (Default: False)
        retry_double_precision (bool): Flag for re-trying eigendecomposition
          with higher precision if lower precision fails due to CuSOLVER
          failure. (Default: True)

    Returns:
        X (Tensor): Inverse root of matrix A.

    """

  # check if matrix is scalar
  if torch.numel(A) == 1:
    alpha = torch.as_tensor(-exponent_multiplier / root)
    return (A + epsilon)**alpha

  # check matrix shape
  if len(A.shape) != 2:
    raise ValueError("Matrix is not 2-dimensional!")
  if A.shape[0] != A.shape[1]:
    raise ValueError("Matrix is not square!")

  if is_diagonal:
    X = matrix_root_diagonal(
        A=A,
        root=root,
        epsilon=epsilon,
        inverse=True,
        exponent_multiplier=exponent_multiplier,
        return_full_matrix=True,
    )
  elif root_inv_method == RootInvMethod.EIGEN:
    X, _, _ = _matrix_root_eigen(
        A=A,
        root=root,
        epsilon=epsilon,
        inverse=True,
        exponent_multiplier=exponent_multiplier,
        retry_double_precision=retry_double_precision,
    )
  elif root_inv_method == RootInvMethod.NEWTON:
    if exponent_multiplier != 1.0:
      raise ValueError(
          f"Exponent multiplier {exponent_multiplier} must be equal to 1 to use"
          "coupled inverse Newton iteration!")

    X, _, termination_flag, _, _ = _matrix_inverse_root_newton(
        A=A,
        root=root,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if termination_flag == NewtonConvergenceFlag.REACHED_MAX_ITERS:
      logging.warning(
          "Newton did not converge and reached maximum number of iterations!")
  else:
    raise NotImplementedError(
        "Root inverse method is not implemented! Specified root inverse method"
        "is " + str(root_inv_method) + ".")

  return X


def matrix_root_diagonal(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    inverse: bool = True,
    exponent_multiplier: float = 1.0,
    return_full_matrix: bool = False,
) -> Tensor:
  """Computes matrix inverse root for a diagonal matrix by taking inverse square
    root of diagonal entries.

    Args:
        A (Tensor): One- or two-dimensional tensor containing either the
          diagonal entries of the matrix or a diagonal matrix.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root.
          (Default: 0.0)
        inverse (bool): Returns inverse root matrix. (Default: True)
        return_full_matrix (bool): Returns full matrix by taking torch.diag of
          diagonal entries. (bool: False)

    Returns:
        X (Tensor): Inverse root of diagonal entries.

    """

  # check order of tensor
  order = len(A.shape)
  if order == 2:
    A = torch.diag(A)
  elif order > 2:
    raise ValueError("Matrix is not 2-dimensional!")

  # check if root is positive integer
  if root <= 0:
    raise ValueError(f"Root {root} should be positive!")

  # compute matrix power
  alpha = exponent_multiplier / root
  if inverse:
    alpha = -alpha

  X = (A + epsilon).pow(alpha)
  return torch.diag(X) if return_full_matrix else X


def _matrix_root_eigen(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    inverse: bool = True,
    exponent_multiplier: float = 1.0,
    make_positive_semidefinite: bool = True,
    retry_double_precision: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
  """Compute matrix (inverse) root using eigendecomposition of symmetric
    positive (semi-)definite matrix.

            A = Q L Q^T => A^{1/r} = Q L^{1/r} Q^T OR A^{-1/r} = Q L^{-1/r} Q^T

    Assumes matrix A is symmetric.

    Args:
        A (Tensor): Square matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root.
          (Default: 0.0)
        inverse (bool): Returns inverse root matrix. (Default: True)
        exponent_multiplier (float): exponent multiplier in the eigen method
          (Default: 1.0)
        make_positive_semidefinite (bool): Perturbs matrix eigenvalues to ensure
          it is numerically positive semi-definite. (Default: True)
        retry_double_precision (bool): Flag for re-trying eigendecomposition
          with higher precision if lower precision fails due to CuSOLVER
            failure. (Default: True)

    Returns:
        X (Tensor): (Inverse) root of matrix. Same dimensions as A.
        L (Tensor): Eigenvalues of A.
        Q (Tensor): Orthogonal matrix consisting of eigenvectors of A.

    """

  # check if root is positive integer
  if root <= 0:
    raise ValueError(f"Root {root} should be positive!")

  # compute matrix power
  alpha = exponent_multiplier / root
  if inverse:
    alpha = -alpha

  # compute eigendecomposition and compute minimum eigenvalue
  try:
    L, Q = torch.linalg.eigh(A)

  except Exception as exception:  # pylint: disable=broad-exception-caught
    if retry_double_precision and A.dtype != torch.float64:
      logger.warning(
          f"Failed to compute eigendecomposition in {A.dtype} precision with "
          f"exception {exception}! Retrying in double precision...")
      L, Q = torch.linalg.eigh(A.double())
    else:
      raise exception

  lambda_min = torch.min(L)

  # make eigenvalues >= 0 (if necessary)
  if make_positive_semidefinite:
    L += -torch.minimum(lambda_min, torch.as_tensor(0.0))

  # add epsilon
  L += epsilon

  # compute inverse preconditioner
  X = Q * L.pow(alpha).unsqueeze(0) @ Q.T

  return X, L, Q


def _matrix_inverse_root_newton(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[Tensor, Tensor, NewtonConvergenceFlag, int, Tensor]:
  """Compute matrix inverse root using coupled inverse Newton iteration.

        alpha <- -1 / p
        X <- 1/c * I
        M <- 1/c^p * A
        repeat until convergence
            M' <- (1 - alpha) * I + alpha * M
            X <- X * M'
            M <- M'^p * M

    where c = (2 |A|_F / (p + 1))^{1/p}. This ensures that |A|_2 <= |A|_F <
    (p + 1) c^p, which guarantees convergence.
    We will instead use z = (p + 1) / (2 * |A|_F).

    NOTE: Exponent multiplier not compatible with coupled inverse Newton
      iteration!

    Args:
        A (Tensor): Matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root.
          (Default: 0.0)
        max_iterations (int): Maximum number of iterations. (Default: 1000)
        tolerance (float): Tolerance. (Default: 1e-6)

    Returns:
        A_root (Tensor): Inverse square root of matrix.
        M (Tensor): Coupled matrix.
        termination_flag (NewtonConvergenceFlag): Specifies convergence.
        iteration (int): Number of iterations.
        error (Tensor): Final error between M and I.

    """

  # initialize iteration, dimension, and alpha
  iteration = 0
  dim = A.shape[0]
  alpha = -1 / root
  identity = torch.eye(dim, dtype=A.dtype, device=A.device)

  # add regularization
  A.add_(identity, alpha=epsilon)

  # initialize matrices
  A_nrm = torch.linalg.norm(A)
  z = (root + 1) / (2 * A_nrm)
  X = z**(-alpha) * identity
  M: Tensor = z * A
  # pyre-fixme[6]: For 1st param expected `Tensor` but got `float`.
  error = torch.dist(M, identity, p=torch.inf)

  # main for loop
  while error > tolerance and iteration < max_iterations:
    iteration += 1
    M_p = M.mul(alpha).add_(identity, alpha=1 - alpha)
    X = X @ M_p
    M = torch.linalg.matrix_power(M_p, root) @ M
    error = torch.dist(M, identity, p=torch.inf)

  # determine convergence flag
  termination_flag = (
      NewtonConvergenceFlag.CONVERGED
      if error <= tolerance else NewtonConvergenceFlag.REACHED_MAX_ITERS)

  return X, M, termination_flag, iteration, error


def compute_matrix_root_inverse_residuals(
    A: Tensor,
    X_hat: Tensor,
    root: int,
    epsilon: float,
    exponent_multiplier: float,
) -> Tuple[Tensor, Tensor]:
  """Compute residual of matrix root inverse for debugging purposes.

        relative error    = ||X - X_hat||_inf / ||X||_inf
        relative residual = ||A X^r - I||_inf

    Args:
        A (Tensor): Matrix of interest.
        X (Tensor): Computed matrix root inverse.
        root (int): Root of interest.
        epsilon (float): Adds epsilon * I to matrix.
        exponent_multiplier (float): Exponent multiplier to be multiplied to the
          numerator of the inverse root.

    Returns:
        absolute_error (Tensor): absolute error of matrix root inverse
        relative_error (Tensor): relative error of matrix root inverse
        residual (Tensor): residual of matrix root inverse

    """

  # check shape of matrix
  if len(A.shape) != 2:
    raise ValueError("Matrix is not 2-dimensional!")
  if A.shape[0] != A.shape[1]:
    raise ValueError("Matrix is not square!")
  if A.shape != X_hat.shape:
    raise ValueError("Matrix shapes do not match!")

  # compute error by comparing against double precision
  X = matrix_inverse_root(
      A.double(),
      root,
      epsilon=epsilon,
      exponent_multiplier=exponent_multiplier)
  relative_error = torch.dist(
      X, X_hat, p=torch.inf) / torch.norm(
          X, p=torch.inf)

  # compute residual
  if exponent_multiplier == 1.0:
    X_invr = torch.linalg.matrix_power(X_hat.double(), n=-root)
  else:
    X_invr, _, _ = _matrix_root_eigen(
        X_hat.double(),
        root=1,
        epsilon=0.0,
        inverse=True,
        make_positive_semidefinite=True,
        exponent_multiplier=root / exponent_multiplier,
    )

  A_reg = A.double() + epsilon * torch.eye(
      A.shape[0], dtype=torch.float64, device=A.device)
  relative_residual = torch.dist(
      X_invr, A_reg, p=torch.inf) / torch.norm(
          A_reg, p=torch.inf)

  return relative_error, relative_residual


def get_dtype_size(dtype: torch.dtype) -> int:
  """Return the size (bytes) of a given data type."""
  return math.ceil(
      (torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits /
      8.0)


def allocate_distributed_tensor(
    shape,
    dtype: torch.dtype,
    device: torch.device,
    device_mesh_ranks: Optional[List[int]] = None,
    use_dtensor: bool = True,
) -> Tensor:
  """Instantiates distributed tensor using Tensor or DTensor.

    Args:
        shape (List[int]): Shape of desired tensor.
        dtype (torch.dtype): DType of desired tensor.
        device (torch.device): Device of desired tensor.
        device_mesh_ranks (Optional[List[int]]): Ranks to use in device mesh of
          desired tensor.
        use_dtensor (bool): Flag for using DTensor. If True and available, uses
          DTensor.  Otherwise, uses Tensor.

    Returns:
        out (Tensor): Desired tensor or DTensor.

    """
  if (ENABLE_DTENSOR and dist.is_initialized() and use_dtensor and
      device_mesh_ranks is not None):
    global _device_mesh_cache  # pylint: disable=global-variable-not-assigned

    key = repr(device_mesh_ranks)
    if key not in _device_mesh_cache:
      _device_mesh_cache[key] = dtensor.DeviceMesh(
          device_type=device.type, mesh=device_mesh_ranks)
    device_mesh = _device_mesh_cache[key]

    return dtensor_zeros(
        shape,
        dtype=dtype,
        device_mesh=device_mesh,
        placements=[dtensor.Replicate()],
    )
  else:
    return torch.zeros(shape, dtype=dtype, device=device)


def use_local_tensor(input_tensor: Tensor) -> Tensor:
  """Uses local tensor if input is a DTensor."""
  return (input_tensor.to_local() if ENABLE_DTENSOR and
          isinstance(input_tensor, dtensor.DTensor) else input_tensor)
