import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from .utils import cat_neighbors_nodes, get_ar_mask
from .constraints import charge_commit, charge_logit_delta, charge_state

class mpnn_sample:
  def sample(self, I):
    """
    I = {
         [[required]]
         'X' = (L,4,3) 
         'mask' = (L,)
         'residue_index' = (L,)
         'chain_idx' = (L,)
         'decoding_order' = (L,)
         
         [[optional]]
         'ar_mask' = (L,L)
         'bias' = (L,21)
         'temperature' = 1.0
         'charge' = {'target','tolerance','lambda_max','values','schedule'}, see mpnn/constraints.py
        }
    """

    key = hk.next_rng_key()
    L = I["X"].shape[0]
    temperature = I.get("temperature",1.0)
    constraint = I.get("charge")

    # prepare node and edge embeddings
    E, E_idx = self.features(I)
    h_V = jnp.zeros((E.shape[0], E.shape[-1]))
    h_E = self.W_e(E)

    ##############
    # encoder
    ##############
    mask_attend = jnp.take_along_axis(I["mask"][:,None] * I["mask"][None,:], E_idx, 1)
    for layer in self.encoder_layers:
      h_V, h_E = layer(h_V, h_E, E_idx, I["mask"], mask_attend)

    # get autoregressive mask  
    ar_mask = I.get("ar_mask",get_ar_mask(I["decoding_order"]))
    
    mask_attend = jnp.take_along_axis(ar_mask, E_idx, 1)
    mask_1D = I["mask"][:,None]
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1 - mask_attend)
    
    h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    h_EXV_encoder = mask_fw[...,None] * h_EXV_encoder

    def fwd(x, t, key, charge_step=None):
      h_EXV_encoder_t = h_EXV_encoder[t] 
      E_idx_t         = E_idx[t]
      mask_t          = I["mask"][t]
      mask_bw_t       = mask_bw[t]      
      h_ES_t          = cat_neighbors_nodes(x["h_S"], h_E[t], E_idx_t)

      ##############
      # decoder
      ##############
      for l,layer in enumerate(self.decoder_layers):
        h_V = x["h_V"][l]
        h_ESV_decoder_t = cat_neighbors_nodes(h_V, h_ES_t, E_idx_t)
        h_ESV_t = mask_bw_t[...,None] * h_ESV_decoder_t + h_EXV_encoder_t
        h_V_t = layer(h_V[t], h_ESV_t, mask_V=mask_t)
        # update
        x["h_V"] = x["h_V"].at[l+1,t].set(h_V_t)

      logits_t = self.W_out(h_V_t)
      x["logits"] = x["logits"].at[t].set(logits_t)

      ##############
      # sample
      ##############
      
      # add bias
      if "bias" in I: logits_t += I["bias"][t]          

      # one gumbel draw for the whole tied group, as upstream ProteinMPNN tied_sample does; averaging
      # a draw per state shrinks it by sqrt(N) and quietly lowers the sampling temperature.
      sampling_logits = logits_t.mean(0, keepdims=True)/temperature

      # net-charge control, on the sampling-scale logits so the lookahead stays a likelihood reweighting
      if constraint is not None:
        charge_delta, x["charge_state"] = charge_logit_delta(x["charge_state"], charge_step, sampling_logits, constraint)
        sampling_logits = sampling_logits + charge_delta

      logits_t = sampling_logits + jax.random.gumbel(key, (1, logits_t.shape[-1]))

      S_t = jax.nn.one_hot(logits_t[...,:20].argmax(-1), 21)

      if constraint is not None:
        x["charge_state"] = charge_commit(x["charge_state"], charge_step, S_t[0], constraint["values"])

      # update
      x["h_S"] = x["h_S"].at[t].set(self.W_s(S_t))
      x["S"]   = x["S"].at[t].set(S_t)
      return x, None
    
    # initial values
    X = {"h_S":    jnp.zeros_like(h_V),
         "h_V":    jnp.array([h_V] + [jnp.zeros_like(h_V)] * len(self.decoder_layers)),
         "S":      jnp.zeros((L,21)),
         "logits": jnp.zeros((L,21))}
    if constraint is not None: X["charge_state"] = charge_state()
    
    # scan over decoding order
    t = I["decoding_order"]
    if t.ndim == 1: t = t[:,None]
    XS = {"t":t, "key":jax.random.split(key,t.shape[0])}
    if constraint is not None: XS["charge"] = constraint["schedule"]
    X = hk.scan(lambda x, xs: fwd(x, xs["t"], xs["key"], xs.get("charge")), X, XS)[0]
    
    out = {"S":X["S"], "logits":X["logits"], "decoding_order":t}
    if constraint is not None: out["charge"] = X["charge_state"]["charge"]
    return out
