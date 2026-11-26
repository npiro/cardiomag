#!/usr/bin/env bash
# Train on any GPU box you can SSH into (cloud VM, lab server, RunPod/Vast pod with SSH).
#
#   remote/run_ssh.sh user@host configs/train/ventricular_cnn.yaml [extra --set overrides]
#
# Steps: rsync code + generated datasets -> set up the env on first use (CUDA wheels) -> train in a
# tmux session (survives disconnects) -> `remote/run_ssh.sh user@host --fetch` copies runs/ back.
set -euo pipefail
HOST="$1"; shift
REMOTE_DIR="${REMOTE_DIR:-~/cardiomag}"
CUDA="${CUDA:-12.4}"
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--fetch" ]]; then
  rsync -avz "$HOST:$REMOTE_DIR/runs/" ./runs_remote/
  exit 0
fi
CONFIG="$1"; shift
OVR="$*"

rsync -avz --exclude .venv --exclude .cache --exclude runs --exclude '.git' --exclude 'data' ./ "$HOST:$REMOTE_DIR/"
rsync -avz ./data/ "$HOST:$REMOTE_DIR/data/"
ssh "$HOST" "cd $REMOTE_DIR && ( [ -x .venv/bin/python ] || bash environment/setup_env.sh --cuda $CUDA ) && \
  tmux new-session -d -s cardiomag \"source .venv/bin/activate && cardiomag-train --config $CONFIG --set device=cuda $OVR 2>&1 | tee train.log\""
echo "started. follow:   ssh $HOST 'tail -f $REMOTE_DIR/train.log'"
echo "results:  remote/run_ssh.sh $HOST --fetch"
