# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the text-translation frontend helpers.

The IIFE in ``text-translation.js`` uses ``Intl.Segmenter`` to count / slice
grapheme clusters. Older browsers lack the API; without a guard the whole
IIFE would throw on load and the text-translation feature would silently
disappear. These tests extract the grapheme helpers from the source file and
drive them in Node under both scenarios.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

JS_PATH = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "static"
    / "js"
    / "text-translation.js"
)

# Extract just the segmenter block so we can run it in isolation without the
# DOM-dependent IIFE surrounding it.
SNIPPET_BOUNDARY_START = "// --- polytalk-grapheme-helpers-start ---"
SNIPPET_BOUNDARY_END = "// --- polytalk-grapheme-helpers-end ---"

NODE_DRIVER = r"""
'use strict';
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const start = src.indexOf(process.argv[4]);
const end = src.indexOf(process.argv[5]);
if (start === -1 || end === -1) {
    console.error('snippet boundaries not found');
    process.exit(2);
}
// Skip past the boundary marker line to the first real line of code.
let block = src.slice(start + process.argv[4].length, end);

// Replace Intl with a fresh object we control, so we can toggle Segmenter.
const IntlShim = {};
if (process.argv[3] === 'with-segmenter') {
    IntlShim.Segmenter = class {
        constructor(_, opts) { this.granularity = opts.granularity; }
        // Split by codepoints, joining combining marks onto the previous
        // segment. Enough to prove the grapheme branch runs differently
        // from the fallback.
        segment(str) {
            const out = [];
            for (const ch of str) {
                const cp = ch.codePointAt(0);
                if (cp >= 0x300 && cp <= 0x36F && out.length) {
                    out[out.length - 1].segment += ch;
                } else {
                    out.push({ segment: ch });
                }
            }
            return out;
        }
    };
}
// else: IntlShim.Segmenter stays undefined → the JS falls back

const run = new Function('Intl', block + '\nreturn {' +
    'count: countGraphemes("abc\\u0301"),' +
    'slice: sliceGraphemes("abcdef", 3),' +
'};');
const output = run(IntlShim);
console.log(JSON.stringify(output));
"""


def _node_available() -> bool:
    return shutil.which("node") is not None


def _run_driver(mode: str, tmp_path: Path) -> dict:
    driver = tmp_path / "driver.js"
    driver.write_text(NODE_DRIVER)
    result = subprocess.run(
        [
            "node",
            str(driver),
            str(JS_PATH),
            mode,
            SNIPPET_BOUNDARY_START,
            SNIPPET_BOUNDARY_END,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(not _node_available(), reason="node not available")
def test_segmenter_block_has_fallback_guard():
    """The JS source guards Intl.Segmenter with a typeof check."""
    source = JS_PATH.read_text()
    assert "typeof Intl.Segmenter" in source
    # When the check fails, we must fall back to str.length / str.slice,
    # not throw.
    assert "return str.length" in source
    assert "return str.slice(0, max)" in source


@pytest.mark.skipif(not _node_available(), reason="node not available")
def test_fallback_counts_utf16_units_when_segmenter_missing(tmp_path):
    """Without Intl.Segmenter, countGraphemes must not throw and returns str.length."""
    payload = _run_driver("without-segmenter", tmp_path)
    # 'abc\u0301' is 4 UTF-16 code units: a, b, c, combining acute accent.
    assert payload["count"] == 4
    assert payload["slice"] == "abc"


@pytest.mark.skipif(not _node_available(), reason="node not available")
def test_grapheme_branch_still_runs_when_segmenter_available(tmp_path):
    """With a (mocked) Intl.Segmenter, countGraphemes uses it, not str.length."""
    payload = _run_driver("with-segmenter", tmp_path)
    # With the mock segmenter, combining accent merges with the previous
    # segment; 'a', 'b', 'c+accent' = 3 graphemes.
    assert payload["count"] == 3
    assert payload["slice"] == "abc"
