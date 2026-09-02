import jax
import jax.numpy as jnp
import flax.core
from models.tiny_model import TinyModel, TinyMemoryConfig

class MemoryBankAdapter:
    """
    Adapter that maintains the Flax state of the TinyModel for benchmarks.
    Drop-in replacement for the previous adapter to avoid changing benchmark scripts.
    """
    def __init__(self, config_path: str = None):
        # We ignore config_path and just use our Tiny config
        self.config = TinyMemoryConfig()
        self.model = TinyModel(config=self.config)
        self.variables = None
        self.rng = jax.random.PRNGKey(42)
        
        # Define JIT-compiled closures
        self._jit_call = jax.jit(lambda v, i, r, w, d: self.model.apply(v, i, r, w, d, mutable=['memory']))
        self._jit_read_only = jax.jit(lambda v, i: self.model.apply(v, i, method=self.model.read_only, mutable=['memory']))
        self._jit_write_only = jax.jit(lambda v, i, e, w: self.model.apply(v, i, e, w, method=self.model.write_only, mutable=['memory']))
        self._jit_decay = jax.jit(lambda v: self.model.apply(v, method=self.model.decay_memory, mutable=['memory']))

    def setup(self):
        self.rng, init_rng = jax.random.split(self.rng)
        dummy_inputs = jnp.zeros((1, self.config.hidden_size), dtype=jnp.float32)
        dummy_read = jnp.zeros((1,))
        dummy_write = jnp.zeros((1,))
        
        def init_fn(module, x, r, w):
            out = module(x, r, w)
            # Dummy h_eos for write/read init
            h_eos = module.get_h_eos(x)
            module.write_only(h_eos, jnp.ones_like(r), w)
            module.read_only(h_eos)
            module.get_v_proj(h_eos)
            return out
            
        self.variables = self.model.init(init_rng, dummy_inputs, dummy_read, dummy_write, method=init_fn)
        
    def reset_memory(self):
        """ Resets only the memory states, keeping learned params intact. """
        self.rng, init_rng = jax.random.split(self.rng)
        dummy_inputs = jnp.zeros((1, self.config.hidden_size), dtype=jnp.float32)
        dummy_read = jnp.zeros((1,))
        dummy_write = jnp.zeros((1,))
        
        def init_fn(module, x, r, w):
            out = module(x, r, w)
            h_eos = module.get_h_eos(x)
            module.write_only(h_eos, jnp.ones_like(r), w)
            module.read_only(h_eos)
            module.get_v_proj(h_eos)
            return out
            
        fresh_vars = self.model.init(init_rng, dummy_inputs, dummy_read, dummy_write, method=init_fn)
        self.variables = {'params': self.variables['params'], 'memory': fresh_vars['memory']}
        
    def load_weights(self, path):
        import flax.serialization
        with open(path, "rb") as f:
            bytes_data = f.read()
            self.variables = flax.serialization.from_bytes(self.variables, bytes_data)
            
    def save_weights(self, path):
        import flax.serialization
        bytes_data = flax.serialization.to_bytes(self.variables)
        with open(path, "wb") as f:
            f.write(bytes_data)

    def advance_time(self, time_steps):
        """ Manually increments global_step in the Flax state to simulate time passing. """
        unfrozen = flax.core.unfreeze(self.variables)
        step = unfrozen['memory']['bank']['global_step']
        unfrozen['memory']['bank']['global_step'] = step + time_steps
        self.variables = flax.core.freeze(unfrozen)
        
    def get_memory_state(self):
        """ Helper to retrieve raw memory state for metric logging. """
        bank_mem = self.variables['memory']['bank']
        return bank_mem['state'], bank_mem['importance'], bank_mem['confidence']

    def __call__(self, inputs, read_prob=None, write_prob=None, deterministic=True):
        if read_prob is None:
            read_prob = jnp.ones((inputs.shape[0],))
        if write_prob is None:
            write_prob = jnp.ones((inputs.shape[0],))
            
        out, new_vars = self._jit_call(self.variables, inputs, read_prob, write_prob, deterministic)
        self.variables = {'params': self.variables['params'], 'memory': new_vars['memory']}
        return out
    
    def read_only(self, inputs):
        out, new_vars = self._jit_read_only(self.variables, inputs)
        self.variables = {'params': self.variables['params'], 'memory': new_vars['memory']}
        return out
        
    def write_only(self, inputs, is_eos, write_prob=None):
        if write_prob is None:
            write_prob = jnp.ones((inputs.shape[0],))
        out, new_vars = self._jit_write_only(self.variables, inputs, is_eos, write_prob)
        self.variables = {'params': self.variables['params'], 'memory': new_vars['memory']}
        
    def decay_memory(self):
        out, new_vars = self._jit_decay(self.variables)
        self.variables = {'params': self.variables['params'], 'memory': new_vars['memory']}
        
    def get_v_proj(self, inputs):
        return self.model.apply(self.variables, inputs, method=self.model.get_v_proj)
        
    def get_h_eos(self, inputs):
        return self.model.apply(self.variables, inputs, method=self.model.get_h_eos)
