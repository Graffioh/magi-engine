#!/usr/bin/env python3
"""Generate a tiny dummy weights file for exercising the mmap loader.

The format is intentionally the simplest thing that works: a flat blob of
little-endian float32 values, no header. The shape is known out-of-band (you
pass it on the command line / hard-code it on the C++ side). Once mmap +
Tensor views are trusted, you can graduate this to a real header (magic, rank,
dims, dtype) without changing how the bytes after the header are read.

Usage:
    python3 tools/gen_dummy_weights.py                 # -> dummy_weights.bin, [2,3] = 1..6
    python3 tools/gen_dummy_weights.py out.bin 4 5     # custom path + shape

Inspect what you wrote:
    hexdump -C dummy_weights.bin
    # 0000  00 00 80 3f  00 00 00 40  ...   (1.0, 2.0, ... as IEEE-754 LE)
"""

import struct
import sys
from math import prod


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "dummy_weights.bin"
    shape = [int(a) for a in sys.argv[2:]] or [2, 3]

    n = prod(shape)
    # Sequential, human-readable values: 1.0, 2.0, ... n.0. Small integers are
    # exactly representable in float32, so the C++ side can compare with tol 0.
    values = [float(i + 1) for i in range(n)]

    # "<" = little-endian, "f" = 32-bit float. struct.pack writes raw bytes in
    # the same layout mmap will hand back to C++ (on a little-endian host).
    blob = struct.pack("<" + "f" * n, *values)

    with open(path, "wb") as fh:
        fh.write(blob)

    print(f"wrote {len(blob)} bytes to {path}  shape={shape}  values={values}")


if __name__ == "__main__":
    main()
