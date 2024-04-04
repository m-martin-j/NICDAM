from __future__ import annotations
"""datamemory"""

__author__ = """Martin Trat"""
__email__ = 'trat@fzi.de'
__copyright__ = "Copyright (c) 2024 Martin Trat"
__license__ = "MIT License"
__version__ = '0.2.6'


import uuid
from typing import Tuple, List, Union
import logging

import pandas as pd

from datamemory.base import DataAccessError


logger = logging.getLogger(__name__)


class Memory:
    def __init__(self):
        pass

    def append(self, addendum):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class DataMemory(Memory):
    """Class for memorizing data, held in a pandas.DataFrame.
    """
    def __init__(
            self,
            batch_size: int=None) -> None:
        """Instantiates a DataMemory object.

        Args:
            batch_size (int, optional): The expected size of data batches. If not provided, it will
                be inferred from the data. Defaults to None.
        """
        self._memory = pd.DataFrame()

        self._registered_memory_pointers = {}  # each pointer is a DataMemoryPointer

        self.batch_size = batch_size

        self._check_parameters()

    def _check_parameters(self) -> None:
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError('batch_size cannot be smaller than 1.')

    def create_memory_pointers(self, pointers: Union[int, list]):
        """Creates a set of pointers.
        # TODO: needs also argument point_to_current_batch --> register_memory_pointer?

        Args:
            pointers (Union[int, list]): The pointers to create. If int provided, pointers new
                pointers are instantiated. If list provided, it must contain one unique name (str)
                per new pointer.

        Returns:
            list: The instantiated pointers.
        """
        ret = []
        if isinstance(pointers, int):  # random-named pointers
            for i in range(pointers):
                ret.append(DataMemoryPointer(memory_ref=self))
        elif isinstance(pointers, list):  # pointer names expected
            for p_name in pointers:
                ret.append(DataMemoryPointer(memory_ref=self,
                                             name=p_name))
        else:
            TypeError('Unexpected type of pointers argument.')
        return ret

    def register_memory_pointer(self, pointer: 'DataMemoryPointer',
                                point_to_current_batch: bool=False) -> None:
        """Registers a new memory pointer.

        Args:
            pointer ('DataMemoryPointer'): The memory pointer.
            point_to_current_batch (bool): If True and memory not empty, an instantiated pointer
                will point to the most recent batch. If False, pointing starts with the next
                incoming batch. Defaults to False.
        """
        if not isinstance(pointer, DataMemoryPointer):
            raise TypeError(f'Provided argument pointer is no DataMemoryPointer.')

        if pointer.name in self._registered_memory_pointers.keys():  # pointer with same name already registered
            raise ValueError(f'Pointer "{pointer.name}" already registered.')
        else:
            pointer.reset()
            self._registered_memory_pointers[pointer.name] = pointer
            if not self.empty and point_to_current_batch:  # point to last batch per default
                # NOTE: if not self.empty then batch_size is already inferred
                recent_batch_idx_range = self._convert_relative_batch_range_2_memory_idx_range((0, 0))
                pointer.point(start=recent_batch_idx_range[0], end=recent_batch_idx_range[1])

    def unregister_memory_pointer(self, pointer: 'DataMemoryPointer') -> None:
        """Unregisters a managed memory pointer.

        Args:
            pointer (DataMemoryPointer): The memory pointer.
        """
        if not isinstance(pointer, DataMemoryPointer):
            raise TypeError(f'Provided argument pointer is no DataMemoryPointer.')

        if pointer not in self._registered_memory_pointers.values():  # pointer not registered
            raise ValueError(f'Pointer "{pointer.name}" not registered.')
        else:
            self._registered_memory_pointers.pop(pointer.name)

    def get_latest_idx(self) -> int:
        if self.empty:
            return None
        else:
            return int(self._memory.index[-1])

    def append(
            self,
            new_data: pd.DataFrame) -> None:
        """Adds more data to memory. DataFrame indices grow with new data. The oldest data row has
        index 0. Each registered pointer's end is updated. If a pointer's start is not set yet, it
        is updated.

        Args:
            new_data (pd.DataFrame): More data.
        """
        if not isinstance(new_data, pd.DataFrame):
            raise TypeError('Addendum new_data needs to be pandas.DataFrame but is'
                            f' {type(new_data)}')
        new_data_n_rows = len(new_data.index)
        if self.batch_size is None:  # infer batch size from first-provided data batch
            self.batch_size = new_data_n_rows
        elif self.batch_size != new_data_n_rows:
            logger.warning(f'Expected batch_size {self.batch_size} but received '
                           f'{new_data_n_rows} data rows.')

        self._memory = pd.concat([self._memory, new_data], ignore_index=True, sort=False)
        augment_len = len(new_data.index)
        logger.debug(f'Augmented memory by {augment_len} rows. Now, memory has {self.length}'
                     ' rows.')

        new_end = self.get_latest_idx()
        for p in self._registered_memory_pointers.values():  # reflect for all registered pointers
            p.point(end=new_end)

    def reset(
            self,
            memorize_most_recent_batch: bool =False,
            hard: bool =False) -> None:
        """Clears the current memory.
        Soft reset: Registered memory pointers will start at last memory index + 1. Memory is kept.
        Hard reset: Actual memory is erased and all pointers are reset.

        Args:
            memorize_most_recent_batch (bool, optional): If True, most recent data batch (self.batch_size data
                rows) are still pointed to after soft reset. Defaults to False.
            hard (bool, optional): If True, saved memory will be dropped entirely and ALL pointers are
                reset. Cannot be True if memorize_most_recent_batch is set to True. Defaults to False.
        """
        # TODO: return deleted memory ranges as DataFrame (copy) on reset?
        reset_msg = 'DataMemory reset'
        if self.empty:
            return

        if hard:  # HARD reset for internal memory and all registered pointers
            reset_msg += ' (hard)'
            if memorize_most_recent_batch:
                # TODO: enable this, e.g. via forget_oldest_n_rows?
                raise ValueError('On hard reset, most recent batch cannot be memorized.')
            self._memory = pd.DataFrame()  # empty
            for p in self._registered_memory_pointers.values():
                p.reset()
        else:  # SOFT reset for all registered pointers
            reset_msg += ' (soft)'
            for p in self._registered_memory_pointers.values():
                p.reset(memorize_most_recent_batch=memorize_most_recent_batch)

        logger.info(reset_msg + '.')

    def forget_oldest_n_rows(self, n: int) -> None:
        """Deletion the n oldest rows.

        Args:
            n (int): The number of rows, counted from the start of the memory, to be deleted.
        """
        if self.empty:
            return

        if n > self.length:
            raise ValueError('Number of rows to be deleted n cannot exceed memory length')
        elif n == self.length:
            self.reset(hard=True)
        else:
            self._memory.drop(self._memory.head(n).index, inplace=True)
            self._memory.reset_index(drop=True, inplace=True)
            logger.info(f'Deleted oldest {n} rows from memory. Now, memory has {self.length}'
                        ' rows.')

            for p in self._registered_memory_pointers.values():  # reflect deletion for all registered pointers
                p_cur_start, p_cur_end = p.get_pointing_bounds()
                if p_cur_start <= (n - 1):
                    if p_cur_end <= (n - 1):  # entire range of pointer deleted
                        p.reset()
                        # NOTE: this should actually never occur, as DataMemoryPointer objects are
                        # always pointing at the end of the internal memory. --> This is part of
                        # the n == self.length case.
                    else:  # only start of pointer changes
                        p.point(start=0, end=(p_cur_end - n))
                else:
                    p.point(start=(p_cur_start - n), end=(p_cur_end - n))

    def get(self, start: int=None, end: int=None) -> pd.DataFrame:
        """Returns the current internal memory based on a specified memory index range (both bounds
        are included). If no range provided, the entire memory is returned.

        Args:
            start (int, optional): The range start. Defaults to None.
            end (int, optional): The range end. Defaults to None.

        Returns:
            pd.DataFrame: The memory.
        """
        if self.empty:
            return pd.DataFrame()  # empty
        elif start is not None or end is not None:
            return self._get_memory_idx_range(start=start, end=end)
        else:  # called without start, end -> entire memory returned
            return self._memory

    def get_batches_relative(self, start: int, end: int) -> pd.DataFrame:
        """Returns the current internal memory based on a specified batch range (both bounds are
        included and are interpreted relative to the latest batch).
        Example: (start,end)=(0,0) refers to the most recent batch, (-1,0) refers to the batch
        before the most recent one and the most recent one.

        Args:
            start (int): The range start.
            end (int): The range end.

        Returns:
            pd.DataFrame: The memory.
        """
        [idx_start, idx_end] = self._convert_relative_batch_range_2_memory_idx_range((start, end))
        return self.get(start=idx_start, end=idx_end)

    @property
    def length(self) -> int:
        """Returns the length (number of rows) of the current memory.

        Returns:
            int: The memory length.
        """
        if self.empty:
            return 0
        _memory = self.get()
        return len(_memory.index)

    def _get_memory_idx_range(self, start, end):
        if start is None:  # TODO: allow smart ranges? No start -> 0, no end or -1 -> latest idx
            raise ValueError('Cannot get memory range without providing start idx.')
        if end is None:
            raise ValueError('Cannot get memory range without providing end idx.')

        if start > end:
            raise ValueError('Memory idx range start cannot be greater than end.')

        if self.empty:
            if not (start == 0 and end == 0):
                raise RuntimeError('Memory being empty not expected?')  # TODO: warn instead?
            return pd.DataFrame()  # empty
        else:
            ret = self._memory.loc[start:end]  # start and end included
            if len(ret.index) < (end - start + 1):
                raise DataAccessError('Length of returned memory idx range smaller than expected.')
            elif len(ret.index) > (end - start + 1):
                raise RuntimeError('Unexpected length of returned memory idx range.')
            return ret

    def _convert_relative_batch_range_2_memory_idx_range(
            self, relative_batch_range: Tuple[int, int]) -> List[int, int]:
        """Converts relative batch-based (--> relative to the latest batch) to row-index-based
        ranges in the memory. example: given a batch size of 3 and 9 elements being contained in
        the memory, relative_batch_range=(0,0) refers to the most recent batch, which is converted
        to [6, 8]. (-1,0) refers to the batch before the most recent one and the most recent one,
        which is converted to [3, 8].

        Args:
            relative_batch_range (Tuple[int, int]): The relative batch range, including both start
                and end.

        Returns:
            List[int, int]: The memory row index range, including both start and end.
        """
        if len(relative_batch_range) != 2:
            raise ValueError('Argument relative_batch_range must have two entries.')
        rel_batch_rg_start, rel_batch_rg_end = relative_batch_range
        if rel_batch_rg_start > rel_batch_rg_end:
            raise ValueError('Relative batch range start cannot be greater than end.')
        latest_memory_idx = self.get_latest_idx()
        idx_start = latest_memory_idx + (rel_batch_rg_start - 1) * self.batch_size + 1
        idx_end = latest_memory_idx + rel_batch_rg_end * self.batch_size
        return [idx_start, idx_end]  # both start and end included

    def get_latest_n_batches(self, n: int):
        if n < 1:
            raise ValueError('n must be greater than or equal to 1.')
        [start, end] = self._convert_relative_batch_range_2_memory_idx_range((1-n,0))
        return self.get(start=start, end=end)

    @property
    def empty(self) -> bool:
        """Checks the memory content.

        Returns:
            bool: True if no memory is held.
        """
        return self._memory.empty


class DataMemoryPointer:
    """Shorthand class for executing Memory methods through a registered memory pointer.
    """
    def __init__(
            self,
            memory_ref: Memory,
            name: str='',
            point_to_current_batch=False) -> None:
        """Instantiates a DataMemoryPointer object.

        Args:
            memory_ref (Memory): The respective Memory object.
            name (str, optional): The name of the memory pointer. If none provided, a random id
                will be generated. Defaults to ''.
            point_to_current_batch (bool): If True and memory not empty, an instantiated pointer
                will point to the most recent batch. If False, pointing starts with the next
                incoming batch. Defaults to False.
        """
        if not isinstance(memory_ref, DataMemory):
            raise TypeError('Provided memory_ref is no DataMemory.')
        self._memory_ref = memory_ref

        if name:
            self.name = name
        else:
            self.name = self._generate_random_ptr_name()

        self.reset(hard=True)

        try:
            self._memory_ref.register_memory_pointer(pointer=self,
                                                     point_to_current_batch=point_to_current_batch)
        except AttributeError:
            raise ValueError('Provided memory_ref is not valid.')

    @staticmethod
    def _generate_random_ptr_name():
        return uuid.uuid4().hex

    def unregister(self) -> None:
        """Unregisters a pointer, rendering it obsolete. (Re-registering is not supported.)
        """
        self._memory_ref.unregister_memory_pointer(pointer=self)
        self.reset(hard=True)

    def point(self, start: int=None, end: int=None) -> None:
        """Sets start and end. Both are understood as row indices and are included.

        Args:
            start (int, optional): The start. Defaults to None.
            end (int, optional): The end. Defaults to None.
        """
        if start is None and end is None:
            raise ValueError('One argument of start and end has to be provided.')

        if start is not None:
            if start < 0:
                raise ValueError('Start cannot be negative.')
            self.start = start
        if end is not None:
            if end < 0:
                raise ValueError('End cannot be negative.')
            if not self._memory_ref.empty and end > self._memory_ref.get_latest_idx() + 1:
                raise ValueError('End cannot exceed current maximum memory idx + 1.')
            self.end = end

        logger.debug(f'Pointer {self.name} has pointing bounds {self.get_pointing_bounds()}')
        if self.start > self.end:
            raise ValueError('Memory pointer start cannot be greater than end.')

    def get_pointing_bounds(self) -> tuple:
        return (self.start, self.end)

    def get(self) -> pd.DataFrame:
        if self.empty:  # memory might not be empty but pointer might be reset (=empty)
            return pd.DataFrame()  # empty
        else:
            return self._memory_ref.get(start=self.start, end=self.end)

    @property
    def memory_ref(self):
        return self._memory_ref

    @property
    def length(self):
        if self._memory_ref.empty:
            return 0
        return len(self.get().index)

    def reset(
            self,
            hard: bool=False,
            memorize_most_recent_batch: bool=False):
        if hard:
            self.start = None
            self.end = None
        else:
            if memorize_most_recent_batch:
                recent_batch_idx_range = \
                    self._memory_ref._convert_relative_batch_range_2_memory_idx_range((0, 0))
                [new_start, new_end] = recent_batch_idx_range \
                    if not self._memory_ref.empty else [0] * 2
            else:  # both pointer start and end point to yet nonexistent data row after last row
                [new_start, new_end] = [self._memory_ref.get_latest_idx() + 1] * 2 \
                    if not self._memory_ref.empty else [0] * 2
            self.point(start=new_start, end=new_end)

    @property
    def empty(self):
        if self._memory_ref.empty:
            _empty = True
        else:
            if None in self.get_pointing_bounds() or (min([self.start, self.end]) > self._memory_ref.get_latest_idx()):
                _empty = True
            else:
                _empty = False
        return _empty
