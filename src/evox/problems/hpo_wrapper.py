__all__ = [
    "HPOMonitor",
    "HPOFitnessMonitor",
    "HPOProblemWrapper",
    "HPOData",
]


import os
import weakref
from abc import ABC
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
from torch import nn

from evox.core import Monitor, Mutable, Problem, Workflow, compile, use_state, vmap

# Env switch for the chunked inner loop (direction 3 of the compile pipeline
# optimization): EOXV_HPO_CHUNK=k compiles a k-step inlined subgraph so Inductor
# can fuse across inner steps inside _hpo_evaluate_loop. k<=1 keeps the stock
# per-step loop. Sweet spot measured at k=8 (per-step 1.03 -> 0.64 ms); larger
# chunks stop helping while compile time grows quadratically (28 steps: 409 s).
_HPO_CHUNK = int(os.environ.get("EOXV_HPO_CHUNK", "0"))

# Env switch for stable wrapper ids (R1 recompile fix): EOXV_HPO_STABLE_ID=1 makes
# the wrapper's `_id_` (which doubles as the Dynamo guard value AND the
# `__hpo_data__` index) a stable per-configuration slot instead of `id(self)`.
# In batch experiments every seed rebuilds the wrapper; a fresh `id(self)` fails
# the `problem._id_ == <int>` guard and forces a full Dynamo re-trace (~23-30 s
# per seed). With stable ids, wrappers built from the same config signature
# (iterations / num_instances / num_repeats / workflow class / state keys)
# share one Dynamo cache entry, so only the first seed pays compilation.
# The slot holds a weakref to its current owner: a live owner keeps the slot
# reserved (no cross-talk between concurrently active wrappers of the same
# signature), and a dead owner frees the slot for reuse.
_STABLE_ID = os.environ.get("EOXV_HPO_STABLE_ID", "0") == "1"
# R10 sensitivity: EOXV_HPO_CR_OFF=1 disables common randomness (randomness="different")
_CR_RAND = "different" if os.environ.get("EOXV_HPO_CR_OFF", "0") == "1" else "same"
_stable_id_slots: Dict[str, Tuple[int, "weakref.ref[Any]"]] = {}
_stable_id_next = 900_000_000


def _stable_hpo_id(wrapper: "HPOProblemWrapper", workflow: "Workflow") -> Optional[int]:
    """Return a stable id for `wrapper` based on its configuration signature,
    or None if the signature cannot be computed (falls back to id(self)).

    Slot stealing: a same-signature newcomer reuses the slot id even if the
    previous owner is still alive (its Dynamo guard closures tend to keep it
    alive, so a strict free-slot policy would hand out a fresh id per seed and
    re-trigger the `problem._id_` guard every time). The newcomer overwrites
    the `__hpo_data__` entry, which is only safe when the previous owner is no
    longer evaluated afterwards — i.e. sequential batch runs. Do NOT enable
    EOXV_HPO_STABLE_ID while concurrently evaluating two same-config wrappers
    (e.g. multi-GPU with threads).
    """
    try:
        sig = (
            wrapper.iterations,
            wrapper.num_instances,
            wrapper.num_repeats,
            type(workflow).__name__,
            id(type(workflow)),
            tuple(wrapper.state_keys) if wrapper.state_keys else (),
        )
        key = repr(sig)
    except Exception:
        return None
    global _stable_id_next
    slot = _stable_id_slots.get(key)
    if slot is None:
        _stable_id_next += 1
        _stable_id_slots[key] = (_stable_id_next, weakref.ref(wrapper))
        return _stable_id_next
    if slot[1]() is wrapper:
        return slot[0]
    # steal: refresh the weakref to the newcomer, keep the id
    _stable_id_slots[key] = (slot[0], weakref.ref(wrapper))
    return slot[0]


def _vmap_vmap_mean_fit_aggregation(info, in_dims, fit: torch.Tensor) -> Tuple[torch.Tensor, int]:
    return torch.mean(fit.movedim(in_dims[0], 0), dim=0, keepdim=True), 0


@torch.library.custom_op("evox::_hpo_vmap_mean_fit_aggregation", mutates_args=())
def _vmap_mean_fit_aggregation(fit: torch.Tensor) -> torch.Tensor:
    return fit.clone()


_vmap_mean_fit_aggregation.register_fake(lambda f: f.new_empty(f.size()))
_vmap_mean_fit_aggregation.register_vmap(_vmap_vmap_mean_fit_aggregation)


@torch.library.custom_op("evox::_hpo_mean_fit_aggregation", mutates_args=())
def _mean_fit_aggregation(fit: torch.Tensor) -> torch.Tensor:
    return fit.clone()


_mean_fit_aggregation.register_fake(lambda f: f.new_empty(f.size()))
_mean_fit_aggregation.register_vmap(lambda info, in_dims, fit: (_vmap_mean_fit_aggregation(fit.movedim(in_dims[0], 0)), 0))


class HPOMonitor(Monitor, ABC):
    """The base class for hyper parameter optimization (HPO) monitors used in `HPOProblem.workflow.monitor`."""

    def __init__(
        self,
        num_repeats: int = 1,
        fit_aggregation: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = _mean_fit_aggregation,
    ):
        super().__init__()
        self.num_repeats = num_repeats
        self.fit_aggregation = fit_aggregation

    def tell_fitness(self) -> torch.Tensor:
        """Get the best fitness found so far in the optimization process that this monitor is monitoring.

        :return: The best fitness so far.
        """
        raise NotImplementedError("`tell_fitness` function is not implemented. It must be overwritten.")


class HPOFitnessMonitor(HPOMonitor):
    """The monitor for hyper parameter optimization (HPO) that records the best fitness found so far in the optimization process."""

    def __init__(
        self,
        num_repeats: int = 1,
        fit_aggregation: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = _mean_fit_aggregation,
        multi_obj_metric: Optional[Callable] = None,
    ):
        """
        Initialize the HPO fitness monitor.

        :param multi_obj_metric: The metric function to use for multi-objective optimization, unused in single-objective optimization.
            Currently we only support "IGD" or "HV" for multi-objective optimization. Defaults to `None`.
        """
        super().__init__(num_repeats, fit_aggregation)
        assert multi_obj_metric is None or callable(multi_obj_metric), (
            f"Expect `multi_obj_metric` to be `None` or callable, got {multi_obj_metric}"
        )
        self.multi_obj_metric = multi_obj_metric
        self.best_fitness = Mutable(torch.tensor(torch.inf))

    def pre_tell(self, fitness: torch.Tensor):
        """Update the best fitness value found so far based on the provided fitness tensor and multi-objective metric.

        :param fitness: A tensor representing fitness values. It can be either a 1D tensor for single-objective optimization or a 2D tensor for multi-objective optimization.

        :raises AssertionError: If the dimensionality of the fitness tensor is not 1 or 2.
        """
        fitness = self.fit_aggregation(fitness) if self.num_repeats > 1 else fitness
        if fitness.ndim == 1:
            # single-objective
            self.best_fitness = torch.min(torch.min(fitness), self.best_fitness)
        else:
            # multi-objective
            self.best_fitness = torch.min(self.multi_obj_metric(fitness), self.best_fitness)

    def tell_fitness(self) -> torch.Tensor:
        """Get the best fitness found so far in the optimization process that this monitor is monitoring.

        :return: The best fitness so far.
        """
        return self.best_fitness


def get_sub_state(state: Dict[str, Any], name: str):
    """Get the sub state from the tuple of states.

    :param state: The tuple of states.

    :return: The sub state.
    """
    prefix_len = len(name) + 1
    state = {k[prefix_len:]: v for k, v in state.items() if k.startswith(name)}
    return state


class HPOData(NamedTuple):
    workflow_step: Callable[[Dict[str, torch.Tensor]], Tuple[Dict[str, torch.Tensor]]]  # workflow_step
    compiled_workflow_step: Callable[[Dict[str, torch.Tensor]], Tuple[Dict[str, torch.Tensor]]]  # compiled_workflow_step
    state_keys: List[str]  # state_keys or param_keys
    buffer_keys: Optional[List[str]]  # optional buffer_keys
    chunked_compiled_step: Optional[Callable[[Dict[str, torch.Tensor]], Tuple[Dict[str, torch.Tensor]]]] = None  # k-step inlined variant


__hpo_data__: Dict[int, HPOData] = {}
# sidecar: id -> owner identity token, used by finalizers to avoid deleting a
# successor's entry after stable-id slot stealing
__hpo_owner_tokens__: Dict[int, object] = {}


class _ChunkBuildRequest:
    """Deferred builder registry for the chunked inner step.

    The compiled custom-op impl cannot capture `self` directly (the op is a
    free function), so the wrapper instance registers a zero-arg callable
    here; _hpo_evaluate_loop picks it up on first real execution.

    IMPORTANT: builders must not capture the wrapper instance strongly —
    that would keep the HPOProblemWrapper (and its compiled graphs / GPU
    buffers) alive forever via this module-level registry, a self-sustaining
    leak across multi-task experiment runs. The registry is therefore keyed
    by object id and stores a weak reference to the instance; entries are
    dropped when the instance dies.
    """

    _registry: Dict[int, Tuple["weakref.ref[Any]", object]] = {}

    @staticmethod
    def register(owner: Any) -> None:
        # key by the wrapper's `_id_` (may be a stable slot id, not id(self))
        # so _resolve_chunk_builder — invoked with the custom-op's id argument —
        # finds this entry
        _key = getattr(owner, "_id_", None) or id(owner)
        _token = object()

        def _cb(_ref, _id=_key, _tok=_token):
            cur = _ChunkBuildRequest._registry.get(_id)
            if cur is not None and cur[1] is _tok:
                _ChunkBuildRequest._registry.pop(_id, None)

        # the SAME ref object carries both the registry entry and the cleanup
        # callback — a separate callback-only ref would itself be garbage and
        # drop the callback
        _ref = weakref.ref(owner, _cb)
        _ChunkBuildRequest._registry[_key] = (_ref, _token)
        owner.__chunk_token__ = _token

    @staticmethod
    def pop(id: int) -> None:
        _ChunkBuildRequest._registry.pop(id, None)

    @staticmethod
    def pop_if_owner(id: int, token: object) -> None:
        cur = _ChunkBuildRequest._registry.get(id)
        if cur is not None and cur[1] is token:
            _ChunkBuildRequest._registry.pop(id, None)


def _resolve_chunk_builder(id: int) -> Optional[Callable[[], Any]]:
    """Look up the chunk builder for the wrapper with the given id.

    Returns a zero-arg callable that builds (if needed) and returns the
    chunked compiled step, or None if the wrapper is gone or has none.
    """
    entry = _ChunkBuildRequest._registry.get(id)
    if entry is None:
        return None
    ref = entry[0]
    wrapper = ref()
    if wrapper is None or getattr(wrapper, "_chunked_compiled_step_", None) is not None:
        return None
    method = getattr(wrapper, "_build_chunked_step", None)

    def _builder():
        if method is not None:
            method()
        return wrapper._chunked_compiled_step_

    return _builder


def _fake_hpo_evaluate_loop(compiling: bool, id: int, iterations: int, state_values: List[torch.Tensor]) -> List[torch.Tensor]:
    return [v.new_empty(v.size()) for v in state_values]


@torch.library.custom_op("evox::_hpo_evaluate_loop", mutates_args=())
def _hpo_evaluate_loop(compiling: bool, id: int, iterations: int, state_values: List[torch.Tensor]) -> List[torch.Tensor]:
    global __hpo_data__
    data = __hpo_data__[id]
    workflow_step, compiled_workflow_step, state_keys, buffer_keys, chunked_compiled_step = data
    if chunked_compiled_step is None and compiling and _HPO_CHUNK > 1 and buffer_keys is None:
        # Lazy build here (not in evaluate): with a warm fx_graph_cache the
        # outer graph is restored from disk and evaluate's Python body is never
        # re-traced, so this impl is the only hook that reliably runs.
        holder = _resolve_chunk_builder(id)
        if holder is not None:
            chunked_compiled_step = holder()
            __hpo_data__[id] = data._replace(chunked_compiled_step=chunked_compiled_step)

    def run_chunked(state: Dict[str, torch.Tensor], n_iter: int) -> Dict[str, torch.Tensor]:
        """Run n_iter steps via the k-step inlined compiled subgraph (plus remainder).

        RNG note: like the per-step compiled loop this path is *not* run-to-run
        deterministic under vmap(randomness=...) — the stock compiled step has
        the same property (verified: same input twice differs). Parity must be
        checked statistically (multi-seed IGD), not bitwise.
        """
        k = _HPO_CHUNK
        for _ in range(n_iter // k):
            state = chunked_compiled_step(state)
        for _ in range(n_iter % k):
            state = compiled_workflow_step(state)
        return state

    if buffer_keys is None:
        state = {k: v.clone() for k, v in zip(state_keys, state_values)}
        if compiling:
            if chunked_compiled_step is not None:
                state = run_chunked(state, iterations)
            else:
                for _ in range(iterations):
                    state = compiled_workflow_step(state)
        else:
            for _ in range(iterations):
                state = workflow_step(state)
        return [state[k] for k in state_keys]
    else:
        param_keys, buffer_keys = state_keys, buffer_keys
        params = {k: v.clone() for k, v in zip(param_keys, state_values)}
        buffers = {k: v.clone() for k, v in zip(buffer_keys, state_values[len(param_keys) :])}
        if compiling:
            if chunked_compiled_step is not None:
                state = run_chunked({**params, **buffers}, iterations)
                params = {k: state[k] for k in param_keys}
                buffers = {k: state[k] for k in buffer_keys}
            else:
                for _ in range(iterations):
                    params, buffers = compiled_workflow_step(params, buffers)
        else:
            for _ in range(iterations):
                params, buffers = workflow_step(params, buffers)
        return [params[k] for k in param_keys] + [buffers[k] for k in buffer_keys]


_hpo_evaluate_loop.register_fake(_fake_hpo_evaluate_loop)


class HPOProblemWrapper(Problem):
    """The problem for hyper parameter optimization (HPO).

    ## Example
    ```python
    algo = SomeAlgorithm(...)
    prob = SomeProblem(...)
    monitor = HPOFitnessMonitor()
    workflow = StdWorkflow(algo, prob, monitor=monitor)
    hpo_prob = HPOProblemWrapper(iterations=..., num_instances=...)
    params = hpo_prob.get_init_params()
    # alter `params` ...
    hpo_prob.evaluate(params) # execute the evaluation
    # ...
    ```
    """

    def __init__(
        self,
        iterations: int,
        num_instances: int,
        workflow: Workflow,
        num_repeats: int = 1,
        copy_init_state: bool = False,
    ):
        """Initialize the HPO problem wrapper.

        :param iterations: The number of iterations to be executed in the optimization process.
        :param num_instances: The number of instances to be executed in parallel in the optimization process, i.e., the population size of the outer algorithm.
        :param workflow: The workflow to be used in the optimization process. Must be wrapped by `core.jit_class`.
        :param num_repeats: The number of times to repeat the evaluation process for each instance. Defaults to 1.
        :param copy_init_state: Whether to copy the initial state of the workflow for each evaluation. Defaults to `True`. If your workflow contains operations that IN-PLACE modify the tensor(s) in initial state, this should be set to `True`. Otherwise, you can set it to `False` to save memory.
        """
        super().__init__()
        assert iterations > 0, f"`iterations` should be greater than 0, got {iterations}"
        assert num_instances > 0, f"`num_instances` should be greater than 0, got {num_instances}"
        self.iterations = iterations
        self.num_instances = num_instances
        self.num_repeats = num_repeats
        self.copy_init_state = copy_init_state
        # check monitor
        monitor = workflow.monitor
        assert isinstance(monitor, HPOMonitor), f"Expect workflow monitor to be `HPOMonitor`, got {type(monitor)}"
        monitor.num_repeats = num_repeats

        # compile workflow steps
        state_step = use_state(workflow.step)

        def repeat_state_step(params: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor]):
            state = {**params, **buffers}
            state = state_step(state)
            return {k: state[k] for k in params.keys()}, {k: state[k] for k in buffers.keys()}

        vmap_state_step = (
            vmap(
                torch.vmap(repeat_state_step, randomness=_CR_RAND),
                randomness="different",
                in_dims=(None, 0),
                out_dims=(None, 0),
            )
            if num_repeats > 1
            else vmap(state_step, randomness=_CR_RAND)
        )
        self._init_params, self._init_buffers = torch.func.stack_module_state([workflow] * self.num_instances)
        if num_repeats > 1:
            self._init_buffers = {k: torch.stack([v] * num_repeats) for k, v in self._init_buffers.items()}
        self._workflow_step_ = vmap_state_step
        self._compiled_workflow_step_ = compile(vmap_state_step, fullgraph=True)

        if type(workflow).init_step == Workflow.init_step:
            # if no init step
            self._workflow_init_step_ = self._workflow_step_
            self._compiled_init_step_ = self._compiled_workflow_step_
        else:
            # otherwise, compile workflow init step
            state_init_step = use_state(workflow.init_step)

            def repeat_state_init_step(params: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor]):
                state = {**params, **buffers}
                state = state_step(state)
                return {k: state[k] for k in params.keys()}, {k: state[k] for k in buffers.keys()}

            vmap_state_init_step = (
                vmap(
                    torch.vmap(repeat_state_init_step, randomness="same"),
                    randomness="different",
                    in_dims=(None, 0),
                    out_dims=(None, 0),
                )
                if num_repeats > 1
                else vmap(state_init_step, randomness="same")
            )
            self._workflow_init_step_ = vmap_state_init_step
            self._compiled_init_step_ = compile(vmap_state_init_step, fullgraph=True)

        if type(workflow).final_step == Workflow.final_step:
            # if no final step
            self._workflow_final_step_ = self._workflow_step_
            self._compiled_final_step_ = self._compiled_workflow_step_
        else:
            # otherwise, compile workflow final step
            state_final_step = use_state(workflow.final_step)

            def repeat_state_final_step(params: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor]):
                state = {**params, **buffers}
                state = state_final_step(state)
                return {k: state[k] for k in params.keys()}, {k: state[k] for k in buffers.keys()}

            vmap_state_final_step = (
                vmap(
                    torch.vmap(repeat_state_final_step, randomness="same"),
                    randomness="different",
                    in_dims=(None, 0),
                    out_dims=(None, 0),
                )
                if num_repeats > 1
                else vmap(state_final_step, randomness="same")
            )
            self._workflow_final_step_ = vmap_state_final_step
            self._compiled_final_step_ = compile(vmap_state_step, fullgraph=True)

        self.state_keys = (list(self._init_params.keys()), list(self._init_buffers.keys()))
        if self.num_repeats == 1:
            self.state_keys = sum(self.state_keys, [])
        global __hpo_data__
        _rid = _stable_hpo_id(self, workflow) if _STABLE_ID else None
        _rid = _rid if _rid is not None else id(self)
        __hpo_data__.pop(_rid, None)  # dead-owner slot may hold stale data
        __hpo_data__[_rid] = HPOData(
            workflow_step=self._workflow_step_,
            compiled_workflow_step=self._compiled_workflow_step_,
            state_keys=self.state_keys if self.num_repeats == 1 else self.state_keys[0],
            buffer_keys=None if self.num_repeats == 1 else self.state_keys[1],
        )
        self._chunked_compiled_step_ = None  # built lazily on first real evaluate
        self._id_ = _rid
        if _HPO_CHUNK > 1 and self.num_repeats == 1:
            _ChunkBuildRequest.register(self)
        # Finalizer cleanup with owner check: under EOXV_HPO_STABLE_ID a later
        # same-signature wrapper may have overwritten our entries in
        # __hpo_data__ / the chunk registry; only delete what is still ours.
        self._finalizer_token_ = object()

        def _cleanup(rid: int, token: object):
            _ChunkBuildRequest.pop_if_owner(rid, token)
            if __hpo_owner_tokens__.get(rid) is token:
                __hpo_owner_tokens__.pop(rid, None)
                __hpo_data__.pop(rid, None)

        weakref.finalize(self, _cleanup, _rid, self._finalizer_token_)
        __hpo_owner_tokens__[_rid] = self._finalizer_token_

        self._stateful_tell_fitness = use_state(monitor.tell_fitness)

    def _build_chunked_step(self):
        """Compile the k-step inlined variant of the inner step (direction 3).

        Returns None; the compiled callable is stored on
        ``self._chunked_compiled_step_`` and published into ``__hpo_data__``.
        Only for num_repeats == 1 flat-state workflows; other configs keep the
        stock loop. Compilation is expensive (~25 s for k=8) so it is deferred
        to the first compiled evaluate call and cached in __hpo_data__.
        """
        if self.num_repeats != 1 or _HPO_CHUNK <= 1:
            return
        if self._chunked_compiled_step_ is not None:
            return

        # Bind the inner step to a local name so the closure below does NOT
        # capture `self`: the compiled callable is stored in the module-level
        # __hpo_data__ and Dynamo's cache, and either would otherwise keep
        # this wrapper (and its GPU buffers) alive forever.
        inner_step = self._compiled_workflow_step_

        def _chunked(state: Dict[str, torch.Tensor]):
            for _ in range(_HPO_CHUNK):
                state = inner_step(state)
            return state

        self._chunked_compiled_step_ = compile(_chunked, fullgraph=True)
        data = __hpo_data__[self._id_]
        __hpo_data__[self._id_] = data._replace(chunked_compiled_step=self._chunked_compiled_step_)

    def evaluate(self, hyper_parameters: Dict[str, nn.Parameter]):
        """
        Evaluate the fitness (given by the internal workflow's monitor) of the batch of hyper parameters by running the internal workflow.

        :param hyper_parameters: The hyper parameters to evaluate.

        :return: The final fitness of the hyper parameters.
        """
        # hyper parameters check
        for k, _ in hyper_parameters.items():
            assert k in self._init_params, (
                f"`{k}` should be a hyperparameter of the workflow, available keys are {self.get_params_keys()}"
            )

        if self.num_repeats > 1:
            if self.copy_init_state:
                params = {k: v.clone() for k, v in self._init_params.items()}
                buffers = {k: v.clone() for k, v in self._init_buffers.items()}
            else:
                params = self._init_params
                buffers = self._init_buffers
            params = {**self._init_params, **hyper_parameters}
            # run the workflow
            if torch.compiler.is_compiling():
                params, buffers = self._compiled_init_step_(params, buffers)
            else:
                params, buffers = self._workflow_init_step_(params, buffers)
            state_values = [params[k] for k in self.state_keys[0]] + [buffers[k] for k in self.state_keys[1]]
            state_values = _hpo_evaluate_loop(torch.compiler.is_compiling(), self._id_, self.iterations - 2, state_values)
            params = {k: v for k, v in zip(self.state_keys[0], state_values)}
            buffers = {k: v for k, v in zip(self.state_keys[1], state_values[len(params) :])}
            if torch.compiler.is_compiling():
                params, buffers = self._compiled_final_step_(params, buffers)
            else:
                params, buffers = self._workflow_final_step_(params, buffers)
            monitor_state = get_sub_state(buffers, "monitor")
            _, fit = vmap(torch.vmap(self._stateful_tell_fitness))(monitor_state)
            return fit[0]
        else:
            state: Dict[str, torch.Tensor] = {**self._init_params, **self._init_buffers}
            if self.copy_init_state:
                state = {k: v.clone() for k, v in state.items()}
            # Override with the given hyper parameters
            state.update(hyper_parameters)
            # run the workflow
            if torch.compiler.is_compiling():
                state = self._compiled_init_step_(state)
            else:
                state = self._workflow_init_step_(state)
            state_values = [state[k] for k in self.state_keys]
            state_values = _hpo_evaluate_loop(torch.compiler.is_compiling(), self._id_, self.iterations - 2, state_values)
            state = {k: v for k, v in zip(self.state_keys, state_values)}
            if torch.compiler.is_compiling():
                state = self._compiled_final_step_(state)
            else:
                state = self._workflow_final_step_(state)
            monitor_state = get_sub_state(state, "monitor")
            _, fit = vmap(self._stateful_tell_fitness)(monitor_state)
            return fit

    def get_init_params(self):
        """Return the initial hyper-parameters dictionary of the underlying workflow."""
        return self._init_params

    def get_params_keys(self):
        return list(self._init_params.keys())
