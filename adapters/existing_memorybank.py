"""
MemoryBankAdapter – Fixed adapter for benchmark experiments.

Fixes applied:
- BUG-P0-014: advance_time() now uses correct path memory['global_step']
- BUG-P0-015: get_memory_state() now uses correct memory paths
- BUG-P0-016: setup() no longer calls get_v_proj / write during init
- BUG-P0-018: reset_memory() now produces truly empty memory state
"""
import jax
import jax.numpy as jnp
import flax.core
from models.tiny_model import TinyModel, TinyMemoryConfig


class MemoryBankAdapter:
    """
    Adapter that maintains the Flax variable state of TinyModel for benchmarks.
    Provides a clean API: setup, reset_memory, write_only, read_only,
    decay_memory, advance_time, get_memory_state, get_v_proj.
    """

    def __init__(self, config_path: str = None):
        # config_path is accepted but ignored; we use TinyMemoryConfig directly.
        self.config  = TinyMemoryConfig()
        self.model   = TinyModel(config=self.config)
        self.variables = None
        self.rng     = jax.random.PRNGKey(42)

        # JIT-compiled model entry points
        self._jit_call = jax.jit(
            lambda v, h, r, w, d: self.model.apply(
                v, h, r, w, d, mutable=['memory']
            )
        )
        self._jit_read_only = jax.jit(
            lambda v, h: self.model.apply(
                v, h, method=self.model.read_only, mutable=['memory']
            )
        )
        self._jit_write_only = jax.jit(
            lambda v, h, e, w: self.model.apply(
                v, h, e, w, method=self.model.write_only, mutable=['memory']
            )
        )
        self._jit_decay = jax.jit(
            lambda v: self.model.apply(
                v, method=self.model.decay_memory, mutable=['memory']
            )
        )

    def setup(self):
        """
        Initialise model parameters with dummy forward pass.
        Memory starts completely empty (all slots EXPIRED, no data written).
        """
        self.rng, init_rng = jax.random.split(self.rng)
        dim   = self.config.hidden_size
        dummy = jnp.zeros((1, dim), dtype=jnp.float32)
        dummy_p = jnp.ones((1,), dtype=jnp.float32)

        # Use a simple init that touches all sub-modules WITHOUT writing to memory
        self.variables = self.model.init(
            init_rng, dummy, dummy_p, dummy_p, False
        )

    def _blank_memory(self) -> dict:
        """Return a fresh, truly empty memory state (all slots EXPIRED)."""
        cfg = self.config
        cap = cfg.memory_capacity
        dim = cfg.memory_dim

        from models.tiny_memory_bank import STATE_EXPIRED
        return {
            'keys':         jnp.zeros((cap, dim),  dtype=jnp.float32),
            'vals':         jnp.zeros((cap, dim),  dtype=jnp.float32),
            'importance':   jnp.zeros((cap,),      dtype=jnp.float32),
            'confidence':   jnp.zeros((cap,),      dtype=jnp.float32),
            'created_at':   jnp.zeros((cap,),      dtype=jnp.int32),
            'last_access':  jnp.zeros((cap,),      dtype=jnp.int32),
            'access_count': jnp.zeros((cap,),      dtype=jnp.int32),
            'state':        jnp.full((cap,), STATE_EXPIRED, dtype=jnp.int32),
            'global_step':  jnp.zeros((),          dtype=jnp.int32),
        }

    def reset_memory(self):
        """
        Reset only the memory state to completely empty.
        Learned parameters are preserved.
        """
        assert self.variables is not None, "Call setup() first."
        self.variables = {
            'params': self.variables['params'],
            'memory': self._blank_memory(),
        }

    def load_weights(self, path: str):
        """Load model weights from a msgpack file."""
        import flax.serialization
        with open(path, 'rb') as f:
            data = f.read()
        self.variables = flax.serialization.from_bytes(self.variables, data)

    def save_weights(self, path: str):
        """Save model weights to a msgpack file."""
        import flax.serialization
        data = flax.serialization.to_bytes(self.variables)
        with open(path, 'wb') as f:
            f.write(data)

    def advance_time(self, time_steps: int):
        """
        Advance the global_step counter to simulate time passing.
        Fix: BUG-P0-014 – path is memory['global_step'], not memory['bank']['global_step'].
        """
        assert self.variables is not None, "Call setup() first."
        mem = dict(self.variables['memory'])
        mem['global_step'] = mem['global_step'] + time_steps
        self.variables = {'params': self.variables['params'], 'memory': mem}

    def get_memory_state(self):
        """
        Return (state, importance, confidence) arrays.
        Fix: BUG-P0-015 – direct path into memory dict.
        """
        assert self.variables is not None, "Call setup() first."
        mem = self.variables['memory']
        return mem['state'], mem['importance'], mem['confidence']

    def get_active_count(self) -> int:
        """Return number of ACTIVE memory slots."""
        from models.tiny_memory_bank import STATE_ACTIVE
        state, _, _ = self.get_memory_state()
        return int(jnp.sum(state == STATE_ACTIVE))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, inputs, read_prob=None, write_prob=None, deterministic=True):
        """Full pipeline: decay → read → fuse."""
        assert self.variables is not None, "Call setup() first."
        if read_prob is None:
            read_prob = jnp.ones((inputs.shape[0],))
        if write_prob is None:
            write_prob = jnp.ones((inputs.shape[0],))

        out, new_vars = self._jit_call(
            self.variables, inputs, read_prob, write_prob, deterministic
        )
        self.variables = {'params': self.variables['params'], 'memory': new_vars['memory']}
        return out

    def read_only(self, inputs):
        """Read from memory without stepping global clock or decaying."""
        assert self.variables is not None, "Call setup() first."
        out, new_vars = self._jit_read_only(self.variables, inputs)
        self.variables = {'params': self.variables['params'], 'memory': new_vars['memory']}
        return out

    def write_only(self, inputs, is_eos, write_prob=None):
        """Write to memory without reading or fusing."""
        assert self.variables is not None, "Call setup() first."
        if write_prob is None:
            write_prob = jnp.ones((inputs.shape[0],))
        _, new_vars = self._jit_write_only(
            self.variables, inputs, is_eos, write_prob
        )
        self.variables = {'params': self.variables['params'], 'memory': new_vars['memory']}

    def decay_memory(self):
        """Trigger decay without any read/write."""
        assert self.variables is not None, "Call setup() first."
        _, new_vars = self._jit_decay(self.variables)
        self.variables = {'params': self.variables['params'], 'memory': new_vars['memory']}

    def get_v_proj(self, inputs):
        """Return v_proj(inputs) – used for expected-value computation in tests."""
        assert self.variables is not None, "Call setup() first."
        return self.model.apply(self.variables, inputs, method=self.model.get_v_proj)

    def get_h_eos(self, inputs):
        """Return inputs as-is (TinyModel inputs are already h_eos vectors)."""
        return inputs
