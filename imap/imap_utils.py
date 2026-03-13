from typing import Optional
import numpy as np
import torch
from sklearn.metrics import (
	silhouette_score,
	davies_bouldin_score,
	calinski_harabasz_score,
)

def _fisher_separability(X: np.ndarray, y: np.ndarray) -> float:
	classes = np.unique(y)
	mu = X.mean(axis=0)
	D = X.shape[1]
	Sw = np.zeros((D, D), dtype=np.float64)
	Sb = np.zeros((D, D), dtype=np.float64)
	for c in classes:
		Xc = X[y == c]
		if Xc.shape[0] < 2:
			continue
		muc = Xc.mean(axis=0)
		Xc0 = Xc - muc
		Sw += Xc0.T @ Xc0
		diff = (muc - mu)[:, None]
		Sb += Xc.shape[0] * (diff @ diff.T)
	num = float(np.trace(Sb))
	den = float(np.trace(Sw)) + 1e-12
	return num / den

def select_head(
	hidden_state: torch.Tensor,
	text_seq_length: int,
	F: int,
	H: int,
	W: int,
	sep_score: str,
	sep_score_threshold: Optional[float] = None,
	topk: int = -1,
	except_k: int = 0,
	text_seq_back: bool = False,
) -> torch.Tensor:
	if hidden_state.ndim != 3:
		raise ValueError(f"hidden_state must be 3D [num_heads, T, D], got shape {tuple(hidden_state.shape)}")

	num_heads, T, D = hidden_state.shape
	expected_visual = F * H * W
	if T - text_seq_length != expected_visual:
		raise ValueError(
			f"Visual token length mismatch: got T={T}, text_seq_length={text_seq_length} -> visual={T - text_seq_length}, "
			f"but expected F*H*W={expected_visual} (F={F},H={H},W={W})."
		)

	key = sep_score.strip().lower()
	valid_keys = {"silhouette", "dbi", "chi", "fisher"}
	if key not in valid_keys:
		raise ValueError(f"sep_score must be one of {sorted(valid_keys)}, got '{sep_score}'.")

	default_thresholds = {
		"silhouette": 0.1,
		"dbi": 10.0,
		"chi": 100.0,
		"fisher": 0.3,
	}
	thr = default_thresholds[key] if sep_score_threshold is None else float(sep_score_threshold)

	y = np.repeat(np.arange(F, dtype=np.int32), H * W)

	vals = np.zeros((num_heads,), dtype=np.float64)
	selected_mask = torch.zeros((num_heads,), dtype=torch.bool, device=hidden_state.device)

	with torch.no_grad():
		if text_seq_back:
			vis_all = hidden_state[:, :-text_seq_length, :]
		else:
			vis_all = hidden_state[:, text_seq_length:, :]
		for h in range(num_heads):
			X = vis_all[h] 
			X_np = X.detach().to("cpu").float().numpy()

			if key == "silhouette":
				n = X_np.shape[0]
				sample_size = min(5000, n)
				val = float(silhouette_score(X_np, y, metric="euclidean", sample_size=sample_size, random_state=0))
				cond = val > thr
			elif key == "dbi":
				val = float(davies_bouldin_score(X_np, y))
				cond = val < thr
			elif key == "chi":
				val = float(calinski_harabasz_score(X_np, y))
				cond = val > thr
			else:
				val = float(_fisher_separability(X_np, y))
				cond = val > thr

			vals[h] = val
			if topk == -1:
				if cond:
					selected_mask[h] = True

	if int(topk) != -1:
		k = int(topk)
		if k <= 0:
			return torch.empty((0,), dtype=torch.long, device=hidden_state.device)
		start = int(except_k) if except_k is not None else 0
		if start < 0:
			start = 0
		if start >= num_heads:
			return torch.empty((0,), dtype=torch.long, device=hidden_state.device)
		k = min(k, num_heads - start)

		if key == "dbi":
			order = np.argsort(vals)
		else:
			order = np.argsort(vals)[::-1].copy() 

		topk_idx = order[start:start + k].copy()
		return torch.as_tensor(topk_idx, dtype=torch.long, device=hidden_state.device)

	return torch.nonzero(selected_mask, as_tuple=False).squeeze(1)

def select_visual_token(
	visual_tokens: torch.Tensor,
	image_query: Optional[torch.Tensor] = None,
	text_key: Optional[torch.Tensor] = None,   
	F: int = 13,
	return_logits: bool = False,
) -> torch.Tensor:

	H_, T, D = visual_tokens.shape

	if image_query is None or text_key is None:
		raise ValueError("For 'qk_matching', image_query and text_key must be provided.")
	Dq = image_query.shape[-1]
	scale = float(Dq) ** 0.5
	logits = torch.einsum('h p d, h t d -> h p t', image_query, text_key) / scale 
	attn_p = logits 
	Hh, Ptot, Tt = attn_p.shape
	if Ptot % F != 0:
		raise ValueError(f"visual tokens {Ptot} must be divisible by F={F} when frame_wise=True.")
	P = Ptot // F
	attn_pf = attn_p.view(Hh, F, P, Tt)      
	attn_tfp = attn_pf.permute(0, 3, 1, 2)   
	idx = attn_tfp.argmax(dim=-1)            
	if return_logits:
		return idx.to(torch.long), attn_p.transpose(1, 2) / (D**0.5)  
	return idx.to(torch.long)
