import tempfile
import time
from pathlib import Path
from typing import Optional

import jax.experimental.host_callback as hcb
import pyarrow as pa


class EvoXVisMonitor:
    """This class serialize data to apache arrow format,
    which can be picked up and used in EvoXVis.

    Parameters
    ----------
    base_filename
        The base filename of the log file,
        the final filename will be ``<base_filename>_<i>.arrow``,
        where i is an incrementing number.
    out_dir
        This directory to write the log file into.
        When set to None, the default directory will be used.
        The default is ``<TEMP_DIR>/evox``,
        on Windows, it's usually ``C:\TEMP\evox``, and on MacOS/Linux/BSDs it's ``/tmp/evox``.
    out_type
        "stream" or "file",
        For more information, please refer to https://arrow.apache.org/docs/python/ipc.html
    batch_size
        The monitor will buffer the data in memory and write out every `batch size`.
        Choose a larger value may improve I/O performance and improve compression ratio,
        if compression is enabled.
        Default to 64.
    compression
        Controls the compression algorithm used when writing to the file.
        Available options are None, "lz4", "zstd",
        "lz4" is extremely fast, with poor compression ratio,
        "zstd" is fast, but also with good compression ratio.
    """

    def __init__(
        self,
        base_filename: str,
        out_dir: str = None,
        out_type: str = "file",
        batch_size: int = 64,
        compression: Optional[str] = None,
    ):
        self.get_time = time.perf_counter_ns
        self.batch_size = batch_size
        self.generation_counter = 0
        self.batch_record = []
        if out_dir is None:
            base_path = Path(tempfile.gettempdir()).joinpath("evox")
        else:
            base_path = Path(out_dir)
        # if the log dir is not created, create it first
        if not base_path.exists():
            base_path.mkdir(parents=True, exist_ok=True)
        # find the next available number
        i = 0
        while base_path.joinpath(f"{base_filename}_{i}.arrow").exists():
            i += 1
        path_str = str(base_path.joinpath(f"{base_filename}_{i}.arrow"))
        self.sink = pa.OSFile(path_str, "wb")
        self.out_type = out_type
        self.compression = compression

        # the schema should be infered at runtime
        # so that we can use fixed size binary which is more efficient
        self.population_size = None
        self.population_dtype = None
        self.fitness_dtype = None

        # the ec_schema is left empty until the first write
        # then we can infer the schema
        self.ec_schema = None
        self.writer = None

        self._reset_batch()

    def _reset_batch(self):
        self.generation = []
        self.timestamp = []

        self.population = []
        self.fitness = []

    def _write_batch(self):
        if len(self.generation) == 0:
            return

        # first time writing the data
        # infer the data schema
        # and create the writer
        if self.ec_schema is None:
            population_byte_len = len(self.population[0])
            fitness_byte_len = len(self.fitness[0])

            self.ec_schema = pa.schema(
                [
                    ("generation", pa.uint64()),
                    ("timestamp", pa.timestamp("ns")),
                    ("population", pa.binary(population_byte_len)),
                    ("fitness", pa.binary(fitness_byte_len)),
                ],
                metadata={
                    "population_size": str(self.population_size),
                    "population_dtype": self.population_dtype,
                    "fitness_dtype": self.fitness_dtype,
                },
            )

            if self.out_type == "file":
                self.writer = pa.ipc.new_file(
                    self.sink,
                    self.ec_schema,
                    options=pa.ipc.IpcWriteOptions(compression=self.compression),
                )

        # actually write the data to disk
        self.writer.write_batch(
            pa.record_batch(
                [
                    self.generation,
                    self.timestamp,
                    self.population,
                    self.fitness,
                ],
                schema=self.ec_schema,
            )
        )
        self._reset_batch()

    def record_pop(self, population, transform=None):
        self.population.append(population.tobytes())
        self.population_size = population.shape[0]
        self.population_dtype = str(population.dtype)
        return population

    def record_fit(self, fitness, transform=None):
        self.generation.append(self.generation_counter)
        self.timestamp.append(self.get_time())
        self.fitness.append(fitness.tobytes())
        self.fitness_dtype = str(fitness.dtype)
        self.generation_counter += 1

        if len(self.fitness) >= self.batch_size:
            self._write_batch()

        return fitness

    def flush(self):
        hcb.barrier_wait()

    def close(self):
        self.flush()
        self._write_batch()
        self.writer.close()
        self.sink.close()
