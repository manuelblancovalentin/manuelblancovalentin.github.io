#!/bin/bash
# Local Jekyll server startup script
# Requires Ruby 3.2+ and dependencies installed via: bundle install

export PATH="/opt/homebrew/opt/ruby@3.2/bin:$PATH"

PORT=${1:-5000}

echo "Starting Jekyll server on http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo ""

bundle exec jekyll serve --port $PORT
