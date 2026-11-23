"""Candidate-conditioned bidirectional Transformer for ComicsPAP Task 1.1.

Place at  src/models/fusion.py  (replaces the previous version; defaults reproduce the old behaviour exactly,
so existing checkpoints and 13_dump_logits.py keep working).

Feature order convention (matches the cache): panels 0-3 = context (original order),
panels 4-8 = candidates 0..4. missing_position m in [0, 4] is the slot of the gap inside
the five-position narrative window.

text_mode (only used when use_text=True):
    "embed"  the OCR-text embedding of every panel is projected and added to the panel representation
             (the original behaviour; the text content acts as an identity of the panel -> easy to memorise)
    "sim"    RELATIONAL text only: the text content is never embedded. Each candidate gets 7 scalar features
             built from the cosine similarity of its OCR text to the context panels' OCR text (similarity to the panel
             right before / right after the gap, max / mean over the context, availability masks). Panel-level
             "has text" flags are still embedded.
    "both"   "embed" + "sim".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SIM_FEATURES = 7


def build_candidate_sequences(ctx: torch.Tensor, cand: torch.Tensor, missing_pos: torch.Tensor) -> torch.Tensor:
    """Insert each candidate at the missing position of the context.

    ctx:         [B, 4, D]   context panels in original relative order
    cand:        [B, K, D]   K candidates
    missing_pos: [B] long    values in 0..4
    returns:     [B, K, 5, D]; for every candidate j, sequence_j has the 4 context panels in
                 order with candidate j placed exactly at index missing_pos.
    """
    B, n_ctx, D = ctx.shape
    K = cand.shape[1]
    L = n_ctx + 1
    pos = torch.arange(L, device=ctx.device)[None, :]            # [1, 5]
    m = missing_pos.view(B, 1)                                    # [B, 1]
    # slots before the gap take ctx[p]; slots after take ctx[p-1]
    ctx_idx = torch.where(pos < m, pos, pos - 1).clamp(0, n_ctx - 1)          # [B, 5]
    gathered = ctx.gather(1, ctx_idx.unsqueeze(-1).expand(B, L, D))           # [B, 5, D]
    is_gap = (pos == m)                                                       # [B, 5]
    seq = gathered.unsqueeze(1).expand(B, K, L, D).clone()                    # [B, K, 5, D]
    gap_mask = is_gap.unsqueeze(1).unsqueeze(-1).expand(B, K, L, D)
    cand_exp = cand.unsqueeze(2).expand(B, K, L, D)
    return torch.where(gap_mask, cand_exp, seq)


def text_similarity_features(text: torch.Tensor, has_text: torch.Tensor, missing_pos: torch.Tensor,
                             n_ctx: int = 4) -> torch.Tensor:
    """Candidate-to-context OCR-text similarity features.

    text [B, N, Dt] raw text embeddings (N = n_ctx + K), has_text [B, N] in {0,1}, missing_pos [B] in 0..n_ctx.
    returns [B, K, 7]: sim_prev, valid_prev, sim_next, valid_next, sim_max, sim_mean, n_valid/n_ctx
    (prev = context panel right before the gap, next = right after it; 0 where a panel or its text is missing).
    """
    B = text.shape[0]
    has = has_text.float()
    ctx_n = F.normalize(text[:, :n_ctx].float(), dim=-1)
    cand_n = F.normalize(text[:, n_ctx:].float(), dim=-1)
    K = cand_n.shape[1]
    valid = has[:, n_ctx:].unsqueeze(2) * has[:, :n_ctx].unsqueeze(1)              # [B, K, n_ctx]
    sims = torch.einsum("bkd,bcd->bkc", cand_n, ctx_n) * valid                      # [B, K, n_ctx]
    m = missing_pos.long()
    prev_idx = (m - 1).clamp(0, n_ctx - 1).view(B, 1, 1).expand(B, K, 1)
    next_idx = m.clamp(0, n_ctx - 1).view(B, 1, 1).expand(B, K, 1)
    has_prev = (m >= 1).float().view(B, 1)
    has_next = (m <= n_ctx - 1).float().view(B, 1)
    v_prev = valid.gather(2, prev_idx).squeeze(2) * has_prev
    v_next = valid.gather(2, next_idx).squeeze(2) * has_next
    s_prev = sims.gather(2, prev_idx).squeeze(2) * has_prev
    s_next = sims.gather(2, next_idx).squeeze(2) * has_next
    n_valid = valid.sum(2)
    s_max = sims.masked_fill(valid == 0, -2.0).max(2).values
    s_max = torch.where(n_valid > 0, s_max, torch.zeros_like(s_max))
    s_mean = sims.sum(2) / n_valid.clamp(min=1.0)
    return torch.stack([s_prev, v_prev, s_next, v_next, s_max, s_mean, n_valid / n_ctx], dim=-1)


class CandidateConditionedFusion(nn.Module):
    def __init__(self, visual_dim=768, text_dim=384, hidden=256, layers=3, heads=8,
                 ff=1024, dropout=0.1, use_text=True, use_visual=True, num_context=4, text_mode="embed"):
        super().__init__()
        assert use_text or use_visual
        assert text_mode in ("embed", "sim", "both")
        self.use_text, self.use_visual = use_text, use_visual
        self.text_mode = text_mode
        self.num_context = num_context
        self.L = num_context + 1
        self.vis_proj = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden))
        embed_text = use_text and text_mode in ("embed", "both")
        self.txt_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden)) if (embed_text or not use_text) else None
        self.has_text_emb = nn.Embedding(2, hidden)       # distinguishes "no text" from zero embedding
        self.pos_emb = nn.Embedding(self.L, hidden)       # narrative position
        self.role_emb = nn.Embedding(2, hidden)           # 0 = context panel, 1 = candidate slot
        self.drop = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(hidden, heads, ff, dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)  # no mask -> bidirectional
        self.norm = nn.LayerNorm(hidden)
        self.scorer = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(hidden, 1))
        self.sim_proj = None
        if use_text and text_mode in ("sim", "both"):
            self.sim_proj = nn.Sequential(nn.Linear(N_SIM_FEATURES, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def embed_panels(self, visual, text, has_text):
        """visual [B,9,Dv], text [B,9,Dt], has_text [B,9] -> [B,9,H]"""
        h = 0
        if self.use_visual:
            h = h + self.vis_proj(visual)
        if self.use_text:
            if self.txt_proj is not None:
                h = h + self.txt_proj(text) * has_text.unsqueeze(-1)
            h = h + self.has_text_emb(has_text.long())
        return h

    def forward(self, visual, text, has_text, missing_pos):
        panels = self.embed_panels(visual, text, has_text)       # [B,9,H]
        ctx, cand = panels[:, :self.num_context], panels[:, self.num_context:]
        B, K = cand.shape[:2]
        seq = build_candidate_sequences(ctx, cand, missing_pos)  # [B,K,5,H]
        idx = torch.arange(self.L, device=seq.device)
        role = (idx[None, :] == missing_pos[:, None]).long()     # [B,5]
        seq = seq + self.pos_emb(idx)[None, None] + self.role_emb(role)[:, None]
        h = self.encoder(self.drop(seq).flatten(0, 1))            # [B*K,5,H]
        h = self.norm(h).view(B, K, self.L, -1)
        gap = h[torch.arange(B, device=h.device), :, missing_pos]  # [B,K,H]
        if self.sim_proj is not None:
            gap = gap + self.sim_proj(text_similarity_features(text, has_text, missing_pos, self.num_context))
        return self.scorer(gap).squeeze(-1)                        # logits [B,K]
