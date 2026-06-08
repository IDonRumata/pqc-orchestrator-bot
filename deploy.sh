#!/usr/bin/env bash
# deploy.sh - run once on the VPS to set up the pqc-orchestrator-bot.
# Usage: bash deploy.sh
# The .env file must already exist at /opt/pqc-orchestrator-bot/.env before running.
set -euo pipefail

REPO="https://github.com/IDonRumata/pqc-orchestrator-bot.git"
DEPLOY_DIR="/opt/pqc-orchestrator-bot"

echo "[1/5] Checking Docker..."
docker --version
docker compose version

echo "[2/5] Cloning / updating repo at $DEPLOY_DIR ..."
if [ -d "$DEPLOY_DIR/.git" ]; then
  cd "$DEPLOY_DIR"
  git fetch origin
  git reset --hard origin/main
else
  git clone "$REPO" "$DEPLOY_DIR"
  cd "$DEPLOY_DIR"
fi

echo "[3/5] Checking .env ..."
if [ ! -f "$DEPLOY_DIR/.env" ]; then
  echo "ERROR: $DEPLOY_DIR/.env not found. Create it from .env.example and fill in secrets."
  exit 1
fi
echo "  .env found."

echo "[4/5] Building and starting containers (detached)..."
docker compose pull db --quiet
docker compose up -d --build --remove-orphans

echo "[5/5] Waiting for bot to become healthy (60s)..."
sleep 15
docker compose ps
docker compose logs --tail=30 bot

echo ""
echo "Deploy complete. Check logs: docker compose -f $DEPLOY_DIR/docker-compose.yml logs -f bot"
