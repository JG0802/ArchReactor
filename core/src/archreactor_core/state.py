# core/src/archreactor_core/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


def wrap32(x: int) -> int:
    """Wrap an integer to 32-bit unsigned."""
    return x & 0xFFFFFFFF


def sign_extend(value: int, bits: int) -> int:
    """Sign-extend 'value' interpreted as 'bits'-bit signed integer to Python int."""
    sign_bit = 1 << (bits - 1)
    mask = (1 << bits) - 1
    v = value & mask
    return (v ^ sign_bit) - sign_bit


class MemoryAlignmentError(ValueError):
    """Raised when a memory access is not properly aligned."""
    pass


@dataclass
class State:
    """
    Minimal CPU state for RV32-like simulation.

    - pc: program counter (byte address)
    - regs: 32 general-purpose registers (x0..x31). x0 is hardwired to 0.
    - mem: byte-addressable memory implemented as sparse dict {addr: byte}.
    """
    pc: int = 0
    regs: List[int] = field(default_factory=lambda: [0] * 32)
    mem: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.regs) != 32:
            raise ValueError("regs must have length 32")
        self.regs[0] = 0  # enforce x0

    # -----------------------
    # Register helpers
    # -----------------------
    def read_reg(self, idx: int) -> int:
        self._check_reg_index(idx)
        # Return 32-bit wrapped value for consistency.
        return wrap32(self.regs[idx])

    def write_reg(self, idx: int, value: int) -> None:
        self._check_reg_index(idx)
        if idx == 0:
            return  # x0 is always 0
        self.regs[idx] = wrap32(value)

    def _check_reg_index(self, idx: int) -> None:
        if not (0 <= idx < 32):
            raise IndexError(f"register index out of range: x{idx}")

    # -----------------------
    # Memory helpers (byte-addressable)
    # -----------------------
    def load_u8(self, addr: int) -> int:
        """Load one byte. Uninitialized bytes are treated as 0."""
        self._check_addr(addr)
        return self.mem.get(addr, 0) & 0xFF

    def store_u8(self, addr: int, value: int) -> None:
        """Store one byte."""
        self._check_addr(addr)
        self.mem[addr] = value & 0xFF

    def load_u32(self, addr: int) -> int:
        """Load 32-bit little-endian word. Address must be 4-byte aligned."""
        if addr % 4 != 0:
            raise MemoryAlignmentError(f"lw address not aligned: 0x{addr:x}")
        b0 = self.load_u8(addr)
        b1 = self.load_u8(addr + 1)
        b2 = self.load_u8(addr + 2)
        b3 = self.load_u8(addr + 3)
        return wrap32(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))

    def store_u32(self, addr: int, value: int) -> None:
        """Store 32-bit little-endian word. Address must be 4-byte aligned."""
        if addr % 4 != 0:
            raise MemoryAlignmentError(f"sw address not aligned: 0x{addr:x}")
        v = wrap32(value)
        self.store_u8(addr, v & 0xFF)
        self.store_u8(addr + 1, (v >> 8) & 0xFF)
        self.store_u8(addr + 2, (v >> 16) & 0xFF)
        self.store_u8(addr + 3, (v >> 24) & 0xFF)

    def _check_addr(self, addr: int) -> None:
        if addr < 0:
            raise ValueError(f"negative memory address: {addr}")