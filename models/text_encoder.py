import flax.linen as nn
import jax.numpy as jnp

class TextEmbedding(nn.Module):
    vocab_size: int = 2000
    embed_dim: int = 32
    
    @nn.compact
    def __call__(self, input_ids):
        """
        Mengubah token ID menjadi vektor embedding.
        
        Args:
            input_ids: array integer dengan shape (batch_size, sequence_length)
                       berisi token dari BPE Tokenizer.
        
        Returns:
            Vektor dense dengan shape (batch_size, sequence_length, embed_dim)
        """
        # Inisialisasi layer embedding JAX/Flax
        embeddings = nn.Embed(
            num_embeddings=self.vocab_size, 
            features=self.embed_dim,
            name="text_embedding_layer"
        )(input_ids)
        
        return embeddings
