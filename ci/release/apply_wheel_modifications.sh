#!/bin/bash
#
# Usage: bash apply_wheel_modifications.sh <new_version>

VERSION=${1}

sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/rmm/__init__.py
sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/setup.py
