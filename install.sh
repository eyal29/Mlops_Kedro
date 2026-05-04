#!/bin/bash
set -e

apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 git \
  && rm -rf /var/lib/apt/lists/*
