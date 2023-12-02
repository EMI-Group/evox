import tempfile
import time
from pathlib import Path
from typing import Optional
import warnings

import jax.experimental.host_callback as hcb
import pyarrow as pa


class EvoXVisMonitor:
    """This class serialize data to apache arrow format,
    which can be picked up and used in EvoXVis.
    The tensors are stored as fixed size binary
    and the dtype is recorded in the metadata.
    """

    def __init__(
        self,
        base_filename: str,
        out_dir: Optional[str] = None,
        out_type: str = "file",
        batch_size: int = 64,
        compression: Optional[str] = None,
    ):
        """
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
            on Windows, it's usually ``C:\\TEMP\\evox``,
            and on MacOS/Linux/BSDs it's ``/tmp/evox``.
        out_type
            "stream" or "file",
            For more information,
            please refer to https://arrow.apache.org/docs/python/ipc.html
        batch_size
            The monitor will buffer the data in memory
            and write out every `batch size`.
            Choose a larger value may improve I/O performance
            and improve compression ratio, if compression is enabled.
            Default to 64.
        compression
            Controls the compression algorithm used when writing to the file.
            Available options are None, "lz4", "zstd",
            "lz4" is extremely fast, with poor compression ratio,
            "zstd" is fast, but also with good compression ratio.
        """

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
        self.comp_alg = compression

        # the schema should be infered at runtime
        # so that we can use fixed size binary which is more efficient
        self.population_size = None
        self.population_dtype = None
        self.fitness_dtype = None

        # the ec_schema is left empty until the first write
        # then we can infer the schema
        self.ec_schema = None
        self.writer = None
        self.is_closed = False

        self.generation = []
        self.fitness = []
        self.population = None
        self.timestamp = None
        self.ref_time = None
        self.duration = []
        self.metrics = None
        self.metric_names = None

    def _write_batch(self, batch_size):
        if len(self.generation) == 0:
            return

        # first time writing the data
        # infer the data schema
        # and create the writer
        if self.ec_schema is None:
            fitness_byte_len = len(self.fitness[0])
            fields = [
                ("generation", pa.uint64()),
                ("fitness", pa.binary(fitness_byte_len)),
            ]
            metadata = {
                "population_size": str(self.population_size),
                "fitness_dtype": self.fitness_dtype,
            }
            if self.population:
                population_byte_len = len(self.population[0])
                fields.append(("population", pa.binary(population_byte_len)))
                metadata["population_dtype"] = self.population_dtype
            if self.timestamp is not None:
                fields.append(("duration", pa.float64()))
                metadata["begin_time"] = str(self.timestamp)
            if self.metrics:
                # store the metric_names
                # so we can fix the order of these keys
                self.metric_names = self.metrics[0].keys()                
                for name in self.metric_names:
                    print(name)
                    fields.append((name, pa.float64()))
                metadata["metrics"] = "_".join(self.metric_names)

            self.ec_schema = pa.schema(
                fields,
                metadata=metadata,
            )

            if self.out_type == "file":
                self.writer = pa.ipc.new_file(
                    self.sink,
                    self.ec_schema,
                    options=pa.ipc.IpcWriteOptions(compression=self.comp_alg),
                )
        # the actual batch size might be different than self.batch_size
        batch = [
            self.generation[:batch_size],
            self.fitness[:batch_size],
        ]
        if self.population:
            batch.append(self.population[:batch_size])
            self.population = self.population[batch_size:]
        if self.timestamp:
            # since timestamp uses sync call
            # timestamp might comtains more elements than fit and pop
            batch.append(self.duration[:batch_size])
            self.duration = self.duration[batch_size:]
        if self.metrics:
            for name in self.metric_names:
                record = []
                for m in self.metrics[:batch_size]:
                    record.append(m[name].item())
                batch.append(record)
            self.metrics = self.metrics[batch_size:]
        # actually write the data to disk
        self.writer.write_batch(
            pa.record_batch(
                batch,
                schema=self.ec_schema,
            )
        )

        self.generation = self.generation[batch_size:]
        self.fitness = self.fitness[batch_size:]

    def record_pop(self, population, transform=None):
        if self.population is None:
            self.population = []
        self.population.append(population.tobytes())
        self.population_dtype = str(population.dtype)

    def record_fit(self, fitness, metrics=None, transform=None):
        self.generation.append(self.generation_counter)
        self.fitness.append(fitness.tobytes())
        self.population_size = fitness.shape[0]
        self.fitness_dtype = str(fitness.dtype)
        self.generation_counter += 1

        batch_size = len(self.fitness)
        if self.population is not None:
            batch_size = min(batch_size, len(self.population))

        if metrics:
            if self.metrics is None:
                self.metrics = []
            self.metrics.append(metrics)

        if batch_size >= self.batch_size:
            self._write_batch(batch_size)

    def record_time(self, transform=None):
        if self.timestamp is None:
            self.timestamp = time.time()
            self.ref_time = time.monotonic()
        self.duration.append(time.monotonic() - self.ref_time)

    def flush(self):
        hcb.barrier_wait()
        batch_size = len(self.fitness)
        self._write_batch(batch_size)

    def close(self, flush=True):
        self.is_closed = True
        if flush:
            self.flush()
        if self.writer is not None:
            self.writer.close()
        self.sink.close()

    def __del__(self):
        if not self.is_closed:
            warnings.warn(
                (
                    "The monitor is not correctly closed. "
                    "Please close the monitor with `close`."
                )
            )
            self.close(flush=False)
