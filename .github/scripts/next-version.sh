#!/usr/bin/env bash
# Prints the version of HEAD: the version tag HEAD already carries, else the last version
# tag reachable from HEAD bumped by the conventional commits since, else v1.0.0.
# On a branch release/vN the major stays N; a breaking change there is an error.
set -euo pipefail

branch="${1:?branch name}"
pattern='v[0-9]*.[0-9]*.[0-9]*'

current=$(git tag --points-at HEAD --list "$pattern" | sort -V | tail -n 1)
if [ -n "$current" ]; then
  echo "$current"
  exit 0
fi

last=$(git describe --tags --abbrev=0 --match "$pattern" 2>/dev/null || true)
if [ -z "$last" ]; then
  next="v1.0.0"
else
  IFS=. read -r major minor patch <<< "${last#v}"
  log=$(git log --format='%s%n%b' "$last..HEAD")
  if grep -qE '^[a-z]+(\([^)]*\))?!:|^BREAKING[ -]CHANGE:' <<< "$log"; then
    next="v$((major + 1)).0.0"
  elif grep -qE '^feat(\([^)]*\))?:' <<< "$log"; then
    next="v$major.$((minor + 1)).0"
  else
    next="v$major.$minor.$((patch + 1))"
  fi
fi

if [[ "$branch" =~ ^release/v([0-9]+)$ ]]; then
  IFS=. read -r next_major _ <<< "${next#v}"
  if [ "$next_major" != "${BASH_REMATCH[1]}" ]; then
    echo "$branch would release $next; a breaking change belongs on master" >&2
    exit 1
  fi
fi
echo "$next"
