# Remote GPU setup — lessons & fast path

Notes for running the Figure 3 marimo notebook (`notebooks/figure3_col_embedding.py`)
on a fresh **vast.ai** GPU instance. Written after doing it the slow way once, so the
next server takes minutes, not an hour.

## Why bother with a remote GPU

- The column-embedding **forward pass** is the bottleneck on a Mac: MPS has no fused
  attention kernels, so it's *minutes*. On an RTX 4090 the same pass is ~**0.07 s / 512
  columns**.
- Data generation (the SCM prior) is CPU-bound and parallel — a many-core box with
  `n_jobs=-1` makes it fast too. Pick a card like a **3090/4090**; filter instances by
  **CPU core count** (it drives generation speed). A100/H100 are overkill here.

## Fast path (fresh instance)

```bash
# 1. Connect (vast.ai key is arena_key, NOT the default id_rsa)
ssh -p <PORT> -i ~/.ssh/arena_key -o IdentitiesOnly=yes root@<IP>

# 2. On the server: clone, install, launch marimo in tmux on port 2718
git clone https://github.com/xiaohan2012/tabicl.git /workspace/tabicl   # public → HTTPS is fine
cd /workspace/tabicl
source /venv/main/bin/activate            # vast PyTorch image: torch+CUDA preinstalled
uv pip install -e ".[notebook]"
tmux new-session -d -s marimo \
  "cd /workspace/tabicl && source /venv/main/bin/activate && \
   marimo edit notebooks/figure3_col_embedding.py --headless --host 127.0.0.1 --port 2718 --no-token"

# 3. From the laptop: tunnel local 2718 → server 2718, then open http://localhost:2718
ssh -N -L 2718:localhost:2718 -p <PORT> -i ~/.ssh/arena_key -o IdentitiesOnly=yes \
    -o ServerAliveInterval=30 root@<IP>
```

Verify: `curl -s -o /dev/null -w '%{http_code}\n' http://localhost:2718/` should print `200`.
Attach to logs: `ssh -p <PORT> ... -t 'tmux attach -t marimo'` (detach with `Ctrl-b d`).

## Gotchas that cost time (avoid these)

| Symptom | Cause | Fix |
|---|---|---|
| `Permission denied (publickey)` | Default `id_rsa` isn't the vast key | Use `-i ~/.ssh/arena_key -o IdentitiesOnly=yes` |
| ssh commands return **no output** | `pkill -f "marimo edit"` matched its **own** ssh command line and killed the shell | Never `pkill` a pattern that matches your invocation; use `tmux kill-session -t marimo` |
| `git push` SSH **times out** | Outbound **port 22 is blocked** on this network | Push via `ssh://git@ssh.github.com:443/<user>/<repo>.git` |
| `git push` HTTPS **rejected** (`workflow` scope) | `gh` OAuth token can't write `.github/workflows/*` | Same fix — push over SSH 443 (key-based, no scope limit) |
| marimo won't bind / page is vast's Jupyter | **Port 8080 is vast's Jupyter** | Run marimo on **2718** |
| Truncated output from `nohup`/`setsid` | fd inheritance + heredoc-in-ssh quoting | Use **tmux**; send scripts via `ssh host 'cat > f' <<'EOF'` (stdin), not remote heredocs |

## Environment facts (vast PyTorch image)

- Python env: `/venv/main` (activate it; use `uv pip install`). System `python3` also on PATH.
- **`/workspace` may not persist** a recycle/destroy — push anything important to git.
- Agent guide lives at `/etc/vast-agents-guide.md`; capabilities at `http://localhost:11111/capabilities`.
- The notebook auto-selects device **CUDA > MPS > CPU** and uses `n_jobs=-1` off macOS.

## Still-todo speedups

- Commit an idempotent `scripts/remote_setup.sh` wrapping the steps above (new server = one command).
- Add `~/.ssh/config` with `ControlMaster auto` / `ControlPersist` to reuse one connection —
  most of the wall-clock was per-command SSH auth + the vast login banner.
- Add on-disk `.npz` caching in the notebook so repeat runs at a given `(n_columns, seq_len, seed)` are instant.
