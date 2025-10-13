"""The browser preprocessing (app/web/js/preprocess.js) must match preprocess() exactly."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


