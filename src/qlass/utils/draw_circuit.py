from typing import Any

import perceval as pcvl
from perceval.components import Processor
from perceval.rendering.circuit import DebugSkin, PhysSkin, SymbSkin
from perceval.rendering.format import Format

_OUTPUT_FORMATS = {
    "mpl": Format.MPLOT,
    "html": Format.HTML,
    "latex": Format.LATEX,
    "text": Format.TEXT,
}

_SKIN_CLASSES = {
    "phys": PhysSkin,
    "symb": SymbSkin,
    "debug": DebugSkin,
}


def _resolve_circuit(circuit_or_processor: Processor | pcvl.Circuit) -> pcvl.Circuit:
    if isinstance(circuit_or_processor, pcvl.Circuit):
        return circuit_or_processor
    if isinstance(circuit_or_processor, Processor):
        return circuit_or_processor.linear_circuit()
    raise TypeError(
        f"Expected a Perceval Processor or Circuit, got {type(circuit_or_processor).__name__}."
    )


def draw_circuit(
    circuit_or_processor: Processor | pcvl.Circuit,
    output_format: str = "mpl",
    skin: str = "phys",
    compact: bool = False,
    save_path: str | None = None,
    backend: str = "perceval",
) -> None:
    """
    Draw a linear optical circuit.

    Wraps Perceval's ``pdisplay`` and ``pdisplay_to_file`` with sensible defaults
    for inspecting photonic circuits outside the VQE optimization loop.

    Args:
        circuit_or_processor (Processor | pcvl.Circuit): Perceval processor or circuit
        output_format (str): One of ``"mpl"``, ``"html"``, ``"latex"``, ``"text"``
        skin (str): One of ``"phys"``, ``"symb"``, ``"debug"``
        compact (bool): If True, use compact display mode
        save_path (str, optional): If provided, save the figure to this file path
        backend (str): Photonic backend to use for rendering. Currently only
            ``"perceval"`` is supported
    """
    if backend != "perceval":
        raise NotImplementedError(
            f"Backend '{backend}' is not supported yet. Only 'perceval' is available."
        )

    if output_format not in _OUTPUT_FORMATS:
        raise ValueError(
            f"Invalid output_format: {output_format!r}. Must be one of {sorted(_OUTPUT_FORMATS)}."
        )
    if skin not in _SKIN_CLASSES:
        raise ValueError(f"Invalid skin: {skin!r}. Must be one of {sorted(_SKIN_CLASSES)}.")

    circuit = _resolve_circuit(circuit_or_processor)
    fmt = _OUTPUT_FORMATS[output_format]
    skin_instance = _SKIN_CLASSES[skin](compact_display=compact)

    if save_path is not None:
        pcvl.pdisplay_to_file(circuit, save_path, output_format=fmt, skin=skin_instance)
    else:
        pcvl.pdisplay(circuit, output_format=fmt, skin=skin_instance)


def resolve_executor_circuit(result: Any) -> Processor | pcvl.Circuit:
    """
    Extract a Perceval Processor or Circuit from an executor return value.

    Args:
        result (Any): Value returned by a VQE executor

    Returns:
        Processor | pcvl.Circuit: Object suitable for :func:`draw_circuit`

    Raises:
        TypeError: If the executor result cannot be visualized as a circuit
    """
    if isinstance(result, (Processor, pcvl.Circuit)):
        return result
    if hasattr(result, "linear_circuit") and callable(result.linear_circuit):
        return result
    raise TypeError(
        "Executor did not return a Perceval Processor or Circuit. "
        "Pass ansatz_fn to draw_ansatz (e.g. ansatz_fn=le_ansatz) or call "
        "draw_circuit directly on the ansatz output."
    )
