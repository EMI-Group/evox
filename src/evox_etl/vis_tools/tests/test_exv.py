"""Tests for :mod:`evox_etl.vis_tools.exv` — the EvoXVision (``.exv``) binary format.

Covers the dtype mapping (``_get_data_type``), the schema builder
(``new_exv_metadata``) for both 1-D and 2-D fitness inputs, and a full
write/read round-trip through :class:`EvoXVisionAdapter` that decodes the raw
binary payload back to the original arrays.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import pytest

from evox_etl.vis_tools.exv import EvoXVisionAdapter, _get_data_type, new_exv_metadata

# exv metadata type string -> numpy dtype (the "plain dict" used by the reader).
EXV_TYPES: Dict[str, np.dtype] = {
    "u8": np.dtype(np.uint8),
    "u16": np.dtype(np.uint16),
    "u32": np.dtype(np.uint32),
    "u64": np.dtype(np.uint64),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}


def _jsonable(obj: Any) -> Any:
    """Recursively convert tuples to lists so JSON-parsed metadata can be compared."""
    if isinstance(obj, tuple):
        return [_jsonable(item) for item in obj]
    if isinstance(obj, list):
        return [_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _jsonable(value) for key, value in obj.items()}
    return obj


def _decode_chunk(chunk: bytes, schema: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Decode one concatenated chunk using a schema's per-field offset/size/shape."""
    decoded: Dict[str, np.ndarray] = {}
    for field in schema["fields"]:
        dtype = EXV_TYPES[field["type"]]
        count = int(np.prod(field["shape"]))
        assert field["size"] == count * dtype.itemsize
        array = np.frombuffer(
            chunk,
            dtype=dtype,
            count=count,
            offset=field["offset"],
        ).reshape(field["shape"])
        decoded[field["name"]] = array
    return decoded


class TestGetDataType:
    """``_get_data_type`` maps supported numpy dtypes to exv strings."""

    @pytest.mark.parametrize("exv_type,np_type", sorted(EXV_TYPES.items()))
    def test_supported_dtypes(self, exv_type: str, np_type: np.dtype) -> None:
        """Each supported numpy dtype maps to its expected exv type string."""
        assert _get_data_type(np_type) == exv_type
        assert _get_data_type(np.dtype(np_type)) == exv_type

    @pytest.mark.parametrize("np_type", [np.bool_, np.int8])
    def test_unsupported_dtypes_raise(self, np_type: type) -> None:
        """Unsupported dtypes (bool, int8) raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            _get_data_type(np_type)
        assert "Unsupported dtype" in str(excinfo.value)


def _assert_schema(schema: Dict[str, Any], population: np.ndarray, fitness: np.ndarray) -> None:
    """Assert one iteration schema matches the population/fitness it was built from."""
    pop_len = len(population.tobytes())
    fit_len = len(fitness.tobytes())
    assert schema["population_size"] == population.shape[0]
    assert schema["chunk_size"] == pop_len + fit_len
    assert schema["chunk_size"] == len(population.tobytes()) + len(fitness.tobytes())

    pop_field, fit_field = schema["fields"]
    assert pop_field["name"] == "population"
    assert pop_field["type"] == _get_data_type(population.dtype)
    assert pop_field["size"] == pop_len
    assert pop_field["offset"] == 0
    assert tuple(pop_field["shape"]) == population.shape

    assert fit_field["name"] == "fitness"
    assert fit_field["type"] == _get_data_type(fitness.dtype)
    assert fit_field["size"] == fit_len
    assert fit_field["offset"] == pop_len
    assert tuple(fit_field["shape"]) == fitness.shape


class TestNewExvMetadata:
    """``new_exv_metadata`` builds correct initial/rest schemas."""

    def test_one_dimensional_fitness_gives_one_objective(self) -> None:
        """A 1-D fitness array yields ``n_objs == 1``."""
        rng = np.random.default_rng(0)
        pop1 = rng.random((6, 4)).astype(np.float32)
        pop2 = rng.random((5, 4)).astype(np.float32)
        fit1 = rng.random(6).astype(np.float32)
        fit2 = rng.random(5).astype(np.float32)

        metadata = new_exv_metadata(pop1, pop2, fit1, fit2)

        assert metadata["version"] == "v1"
        assert metadata["n_objs"] == 1
        _assert_schema(metadata["initial_iteration"], pop1, fit1)
        _assert_schema(metadata["rest_iterations"], pop2, fit2)

    def test_two_dimensional_fitness_gives_n_objectives(self) -> None:
        """A 2-D fitness array yields ``n_objs == fitness.shape[1]``."""
        rng = np.random.default_rng(1)
        pop1 = rng.random((6, 4)).astype(np.float32)
        pop2 = rng.random((5, 4)).astype(np.float32)
        fit1 = rng.random((6, 3)).astype(np.float64)
        fit2 = rng.random((5, 3)).astype(np.float64)

        metadata = new_exv_metadata(pop1, pop2, fit1, fit2)

        assert metadata["n_objs"] == fit1.shape[1] == 3
        _assert_schema(metadata["initial_iteration"], pop1, fit1)
        _assert_schema(metadata["rest_iterations"], pop2, fit2)
        # Different dtypes for population vs fitness are encoded independently.
        assert metadata["initial_iteration"]["fields"][0]["type"] == "f32"
        assert metadata["initial_iteration"]["fields"][1]["type"] == "f64"

    def test_chunk_size_is_population_plus_fitness_bytes(self) -> None:
        """chunk_size equals population bytes + fitness bytes for both schemas."""
        rng = np.random.default_rng(2)
        pop1 = rng.random((7, 2)).astype(np.float32)
        pop2 = rng.random((4, 2)).astype(np.float32)
        fit1 = rng.random((7, 2)).astype(np.float32)
        fit2 = rng.random((4, 2)).astype(np.float32)

        metadata = new_exv_metadata(pop1, pop2, fit1, fit2)

        assert metadata["initial_iteration"]["chunk_size"] == pop1.nbytes + fit1.nbytes
        assert metadata["rest_iterations"]["chunk_size"] == pop2.nbytes + fit2.nbytes
        assert metadata["initial_iteration"]["population_size"] == 7
        assert metadata["rest_iterations"]["population_size"] == 4


class TestEvoXVisionRoundTrip:
    """Full write -> read round-trip of an exv file with mixed population sizes."""

    def test_round_trip(self, tmp_path) -> None:
        """Header + initial chunk + several rest chunks survive a round trip."""
        rng = np.random.default_rng(42)

        pop1 = rng.random((6, 4)).astype(np.float32)
        fit1 = rng.random((6, 2)).astype(np.float32)
        # Rest iterations use a *different* population size than the initial one.
        rest_pops = [rng.random((4, 4)).astype(np.float32) for _ in range(3)]
        rest_fits = [rng.random((4, 2)).astype(np.float32) for _ in range(3)]

        metadata = new_exv_metadata(pop1, rest_pops[0], fit1, rest_fits[0])

        exv_path = tmp_path / "run.exv"
        adapter = EvoXVisionAdapter(exv_path, buffering=0)
        adapter.set_metadata(metadata)
        adapter.write_header()
        adapter.write(pop1.tobytes(), fit1.tobytes())
        for pop, fit in zip(rest_pops, rest_fits):
            adapter.write(pop.tobytes(), fit.tobytes())
        adapter.flush()
        adapter.writer.close()

        data = exv_path.read_bytes()

        # --- header ------------------------------------------------------
        assert data[:4] == b"\x65\x78\x76\x31"
        metadata_len = int.from_bytes(data[4:8], byteorder="little", signed=False)
        expected_metadata_bytes = json.dumps(metadata).encode("utf-8")
        assert metadata_len == len(expected_metadata_bytes)

        parsed = json.loads(data[8 : 8 + metadata_len].decode("utf-8"))
        assert parsed == _jsonable(metadata)
        assert parsed["version"] == "v1"
        assert parsed["n_objs"] == 2
        assert parsed["initial_iteration"]["chunk_size"] == pop1.nbytes + fit1.nbytes
        assert parsed["rest_iterations"]["chunk_size"] == rest_pops[0].nbytes + rest_fits[0].nbytes

        # --- binary body -------------------------------------------------
        body = data[8 + metadata_len :]
        init_schema = parsed["initial_iteration"]
        rest_schema = parsed["rest_iterations"]

        # Initial chunk.
        init_chunk = body[: init_schema["chunk_size"]]
        decoded_init = _decode_chunk(init_chunk, init_schema)
        assert np.array_equal(decoded_init["population"], pop1)
        assert np.array_equal(decoded_init["fitness"], fit1)

        # Rest chunks.
        pos = init_schema["chunk_size"]
        for expected_pop, expected_fit in zip(rest_pops, rest_fits):
            chunk = body[pos : pos + rest_schema["chunk_size"]]
            assert len(chunk) == rest_schema["chunk_size"]
            decoded = _decode_chunk(chunk, rest_schema)
            assert np.array_equal(decoded["population"], expected_pop)
            assert np.array_equal(decoded["fitness"], expected_fit)
            pos += rest_schema["chunk_size"]

        assert pos == len(body)
        assert body == b"".join(
            [pop1.tobytes(), fit1.tobytes()]
            + [payload for pop, fit in zip(rest_pops, rest_fits) for payload in (pop.tobytes(), fit.tobytes())]
        )
