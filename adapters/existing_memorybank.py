"""
MemoryBankAdapter – PyTorch adapter for benchmark experiments.
"""
import torch
import numpy as np
from models.tiny_model import TinyModel, TinyMemoryConfig
from models.tiny_memory_bank import STATE_ACTIVE, STATE_EXPIRED, STATE_DORMANT


class MemoryBankAdapter:
    """
    Adapter that maintains the state of TinyModel for benchmarks (PyTorch Version).
    Provides a clean API: setup, reset_memory, write_only, read_only,
    decay_memory, advance_time, get_memory_state, get_v_proj.
    """

    def __init__(self, config_path: str = None):
        self.config = TinyMemoryConfig()
        self.model = TinyModel(config=self.config)

    def setup(self):
        """Initialise model parameters and ensure blank memory state."""
        self.reset_memory()

    def reset_memory(self):
        """Reset memory state to completely empty."""
        self.model.bank.load_memory_state(self.model.bank.empty_memory_state())

    def load_weights(self, path: str):
        """Load model weights from a file."""
        self.model.load_state_dict(torch.load(path, map_location='cpu'))

    def save_weights(self, path: str):
        """Save model weights to a file."""
        torch.save(self.model.state_dict(), path)

    def advance_time(self, time_steps: int):
        """Advance the global_step counter to simulate time passing."""
        self.model.bank.global_step[0] += time_steps

    def get_memory_state(self):
        """Return (state, importance, confidence) as NumPy arrays."""
        bank = self.model.bank
        state = bank.mem_state.detach().cpu().numpy()
        importance = bank.mem_importance.detach().cpu().numpy()
        confidence = bank.mem_confidence.detach().cpu().numpy()
        return state, importance, confidence

    def get_active_count(self) -> int:
        """Return number of ACTIVE memory slots."""
        state, _, _ = self.get_memory_state()
        return int(np.sum(state == STATE_ACTIVE))

    def __call__(self, inputs, read_prob=None, write_prob=None, deterministic=True):
        """Full pipeline: decay → read → fuse → decode."""
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        if read_prob is not None and isinstance(read_prob, np.ndarray):
            read_prob = torch.from_numpy(read_prob).float()
        if write_prob is not None and isinstance(write_prob, np.ndarray):
            write_prob = torch.from_numpy(write_prob).float()

        out = self.model(inputs, read_prob=read_prob, write_prob=write_prob, deterministic=deterministic)
        return out.detach().cpu().numpy()

    def read_only(self, inputs):
        """Read from memory without stepping global clock or decaying."""
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        out = self.model.read_only(inputs)
        return out.detach().cpu().numpy()

    def write_only(self, inputs, is_eos, write_prob=None):
        """Write to memory without reading or fusing."""
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        if isinstance(is_eos, np.ndarray):
            is_eos = torch.from_numpy(is_eos).float()
        if write_prob is None:
            write_prob = torch.ones(inputs.shape[0], device=inputs.device)
        elif isinstance(write_prob, np.ndarray):
            write_prob = torch.from_numpy(write_prob).float()

        idx = self.model.write_only(inputs, is_eos, write_prob)
        return idx.detach().cpu().numpy() if idx is not None else None

    def decay_memory(self):
        """Trigger decay without any read/write."""
        eff_R = self.model.decay_memory()
        return eff_R.detach().cpu().numpy()

    def get_v_proj(self, inputs):
        """Return v_proj(inputs) as NumPy array."""
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        out = self.model.get_v_proj(inputs)
        return out.detach().cpu().numpy()

    def get_h_eos(self, inputs):
        return inputs
