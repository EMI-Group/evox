"""This module helps serialize data to EvoXVision storage format (exv).

| Magic Number         | Header Length | Metadata               | Initial Iteration Data | Binary Data   |
| -------------------- | ------------- | ---------------------- | ---------------------- | ------------- |
| 0x65787631 (4 bytes) | u32 (4 bytes) | JSON encoded (n bytes) | binary data            | binary data   |

The numbers are stored in little-endian format.
The metadata is a JSON utf-8 encoded string, which contains the schema of the binary data.
The format of the metadata is as follows:
```json
{
    "version": "v1",
    "n_objs": "<number>",
    "initial_iteration": {
        "population_size": "<number>",
        "chunk_size": "<number>",
        "fields": [
            {
                "name": "<field name>",
                "type": "<type>",
                "size": "<number>",
                "offset": "<number>",
                "shape": ["<number>"]
            }
        ]
    },
    "rest_iterations": {
        "population_size": "<number>",
        "chunk_size": "<number>",
        "fields": [
            {
                "name": "<field name>",
                "type": "<type>",
                "size": "<number>",
                "offset": "<number>",
                "shape": ["<number>"]
            }
        ]
    }
}
```

where <type> represents the data type of the field, available types are:
- "u8", "u16", "u32", "u64",
- "i16", "i32", "i64",
- "f16", "f32", "f64"
The size and offset are in bytes.

```{note}
The magic number is used to identify the file format.
0x65787631 is the byte code for "exv1".
The binary data blob is a sequence of binary data chunks.
In EvoX, the algorithm is allowed to have a different behavior in the first iteration (initialization phase),
which can have a different chunk size than the rest of the iterations.
Therefore it contains two different schemas for the initial iteration and the rest of the iterations.
```
"""

import json
from pathlib import Path
from typing import Union

import numpy as np


def _get_data_type(dtype):
    if dtype == np.uint8:
        return "u8"
    elif dtype == np.uint16:
        return "u16"
    elif dtype == np.uint32:
        return "u32"
    elif dtype == np.uint64:
        return "u64"
    elif dtype == np.int16:
        return "i16"
    elif dtype == np.int32:
        return "i32"
    elif dtype == np.int64:
        return "i64"
    elif dtype == np.float16:
        return "f16"
    elif dtype == np.float32:
        return "f32"
    elif dtype == np.float64:
        return "f64"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def new_exv_metadata(
    population1: np.ndarray,
    population2: np.ndarray,
    fitness1: np.ndarray,
    fitness2: np.ndarray,
):
    """Takes the input of the populaton and fitness from the first two iterations,
    and returns the schema for exv file format."""
    initial_population_size, dim = population1.shape
    if fitness1.ndim == 1:
        n_objs = 1
    else:
        n_objs = fitness1.shape[1]
    initial_population_byte_len = len(population1.tobytes())
    initial_fitness_byte_len = len(fitness1.tobytes())
    initial_iteration = {
        "population_size": initial_population_size,
        "chunk_size": initial_population_byte_len + initial_fitness_byte_len,
        "fields": [
            {
                "name": "population",
                "type": _get_data_type(population1.dtype),
                "size": initial_population_byte_len,
                "offset": 0,
                "shape": population1.shape,
            },
            {
                "name": "fitness",
                "type": _get_data_type(fitness1.dtype),
                "size": initial_fitness_byte_len,
                "offset": initial_population_byte_len,
                "shape": fitness1.shape,
            },
        ],
    }

    population_size, dim = population2.shape
    population_byte_len = len(population2.tobytes())
    fitness_byte_len = len(fitness2.tobytes())
    rest_iterations = {
        "population_size": population_size,
        "chunk_size": population_byte_len + fitness_byte_len,
        "fields": [
            {
                "name": "population",
                "type": _get_data_type(population2.dtype),
                "size": population_byte_len,
                "offset": 0,
                "shape": population2.shape,
            },
            {
                "name": "fitness",
                "type": _get_data_type(fitness2.dtype),
                "size": fitness_byte_len,
                "offset": population_byte_len,
                "shape": fitness2.shape,
            },
        ],
    }

    metadata = {
        "version": "v1",
        "n_objs": n_objs,
        "initial_iteration": initial_iteration,
        "rest_iterations": rest_iterations,
    }
    return metadata


class EvoXVisionAdapter:
    """EvoXVisionAdapter is a class that streams evolutionary optimization data to an exv file.
    The exv file format is a binary format that created specifically for the evolutionary optimization data.
    The format is designed to be efficient for both stream reading and writing data, while being able to randomly access data at any iteration.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        buffering: int = 0,
    ):
        """
        Create a new EvoXVisionAdapter instance, which writes data to an exv file.
        To automatically inference the data schema, the EvoXVisionAdapter requires 2 iterations of data,
        therefore it will only start to write data after the 2 iterations of the optimization loop are completed.

        :param file_path: The path to the exv file
        :param buffering: The buffer size to use for file operations, passed directly to the `open()` function.
            The default is `0`, which disables buffering (unbuffered mode).

        ```{notes}
        * Buffering affects how data is written to the file to minimize system call overhead.
        However, from a filesystem perspective, operations are always be considered buffered.
        * Disabling buffering (`buffering=0`) is often recommended in scenarios where system call overhead
        is not the bottleneck, as it ensures data is immediately written without delay.
        * When buffering is enabled, it may be necessary to call `flush()` explicitly to guarantee that all
        data is written to the file.
        ```
        """
        self.writer = open(file_path, "wb", buffering=buffering)
        self.metadata = None
        self.header_written = False

    def _write_magic_number(self):
        self.writer.write(b"\x65\x78\x76\x31")

    def _write_metedata(self, metadata):
        metadata_bin = json.dumps(metadata).encode(encoding="utf-8")
        metadata_len = len(metadata_bin).to_bytes(4, byteorder="little", signed=False)
        self.writer.write(metadata_len)
        self.writer.write(metadata_bin)

    def set_metadata(self, metadata):
        self.metadata = metadata

    def write_header(self):
        """Write the header of the exv file."""
        assert self.metadata is not None, "Metadata must be set before writing the header."
        self._write_magic_number()
        self._write_metedata(self.metadata)
        self.header_written = True

    def write(self, *fields):
        """Stream data to the exv file.
        Depending on the `buffering` parameter, the data may not be written immediately.
        """
        assert self.header_written, "Header must be written before writing data."
        self.writer.writelines(fields)

    def flush(self):
        """Flush the internal buffer to the file."""
        if self.writer:
            self.writer.flush()
