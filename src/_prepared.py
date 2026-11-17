"""A kernel invocation whose inputs are already resident on the device.

Benchmarks must time the kernel, not the host-side padding and PCIe copies
around it. On a fast GPU those copies dominate: the kernel speeds up ~20x from
a T1000 to an H100 while PCIe does not, so an end-to-end timing would measure
the bus. Every backend therefore exposes a `prepare_*` function that does the
validation, padding and host-to-device staging once and returns one of these.

    call = prepare_flash_decode_pallas(...)
    out = call.run()        # launches the kernel on device-resident inputs
    call.sync(out)          # blocks until the device has finished
    host = call.to_numpy(out)

    host = call()           # all three, for ordinary use

The public entry points (ops.flash_decode, flash_decode_pallas, ...) are
`prepare_*(...)()`, so the path a benchmark times is the path the tests check.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional


def _no_sync(_out: Any) -> None:
    return None


def _identity(out: Any) -> Any:
    return out


@dataclass
class PreparedCall:
    #: Launch the computation on staged inputs; returns device-side outputs.
    run: Callable[[], Any]
    #: Block until the outputs of run() are complete.
    sync: Callable[[Any], None] = _no_sync
    #: Copy run()'s outputs to host NumPy (kept out of any timed region).
    to_numpy: Callable[[Any], Any] = _identity
    #: Compiled IR text, where inspecting it is meaningful (anti-cheat).
    lowered_text: Optional[Callable[[], str]] = None

    def __call__(self) -> Any:
        out = self.run()
        self.sync(out)
        return self.to_numpy(out)
