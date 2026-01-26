## Arch-Reactor Core

The Core is a **pure-Python** RISC-V architecture simulator library.
At the moment, it implements **Stage 1: a non-pipelined (sequential) CPU** only.

### Goals
- Provide a clear, readable CPU execution model for education
- Keep the structure ready for incremental expansion (pipeline, hazards, performance)
- Prioritize correctness and explainability over speed

### Scope
- ISA: RISC-V RV32I (subset)
- Supported instructions: `addi`, `add`, `sub`, `lw`, `sw`, `beq`, `bne`, `jal`
- State: PC, 32 registers, byte-addressable memory
- Register x0 must always remain 0
- Each step produces a JSON-serializable execution frame

### Directory Layout
```
core/
  src/archreactor_core/
    __init__.py      # public package API
    state.py         # CPU state (PC/registers/memory)
    instruction.py   # instruction data model
    parser.py        # assembly parser
    semantics.py     # instruction semantics (sequential)
    effects.py       # execution frame structure
    simulator.py     # step/run orchestration
    exceptions.py    # explicit error types
  tests/
    test_state.py
    test_parser.py
    test_addi.py
```

### Development Principles
- Prefer readable, explicit code
- Do not mix pipeline logic into Stage 1
- Write tests for every core behavior
- Raise explicit exceptions for invalid behavior

### Roadmap (Reference)
- Stage 2: 5-Stage Pipeline CPU
- Stage 3: Hazard handling (stalls/forwarding)
- Stage 4: Performance analysis (cycles, CPI)
- Stage 5: ISA extensions (RV32M, optional branch behavior)
