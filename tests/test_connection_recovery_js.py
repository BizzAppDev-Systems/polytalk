"""Regression tests for the frontend connection-recovery logic.

Drives PolyTalkApp._scheduleReconnect() end-to-end in Node.js (no browser
required) and verifies the constructor initializes the reconnect state
fields that _scheduleReconnect() depends on.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = (
    Path(__file__).resolve().parent.parent / "app" / "static" / "js" / "polytalk-app.js"
)


def _node_supports_app_syntax() -> bool:
    """polytalk-app.js uses modern syntax (optional chaining); Node 14+ required."""
    if shutil.which("node") is None:
        return False
    probe = "const x = { a: null }; process.exit(x.a?.b === undefined ? 0 : 1);"
    try:
        return subprocess.run(["node", "-e", probe]).returncode == 0
    except subprocess.SubprocessError:
        return False


HAS_NODE_APP_SYNTAX = _node_supports_app_syntax()

CONSTRUCTOR_FIELDS = [
    "this.wsConnectionState = 'connected';",
    "this.reconnectAttempts = 0;",
    "this.reconnectTimer = null;",
    "this._lastSavedLang = null;",
    "this._lastSavedMode = null;",
    "this._reconnectMaxAttempts = 7;",
    "this._reconnectBackoffMs = [",
]

NODE_DRIVER = r"""
'use strict';
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');

// Evaluate the app script and grab the class without running the DOM-heavy
// constructor (the class is not exported; append an alias in our own scope).
const scopeFn = new Function(
    'window', 'document', 'navigator', 'localStorage', 'console', 'AudioRecorder',
    src + '\nreturn PolyTalkApp;'
);
class FakeRecorder {
    constructor() { this.isRecording = false; this.isPaused = false; }
    pause() { this.isPaused = true; }
    stop() { this.isRecording = false; }
}
const PolyTalkApp = scopeFn({}, {}, { onLine: true },
    { getItem: () => null, setItem: () => {} }, console, FakeRecorder);

function makeApp() {
    const app = Object.create(PolyTalkApp.prototype);
    app.audioRecorder = new FakeRecorder();
    app.wsConnectionState = 'connected';
    app.isOnline = true;
    app.reconnectAttempts = 0;
    app.reconnectTimer = null;
    app._lastSavedLang = null;
    app._lastSavedMode = null;
    app._reconnectMaxAttempts = 7;
    app._reconnectBackoffMs = [2000, 4000, 8000, 12000, 15000, 15000, 15000];
    app.isStreaming = true;
    app.isPaused = false;
    app.reconnects = [];
    app.statuses = [];
    app.uiStates = [];
    app.uiText = (m) => m;
    app._showConnectionStatus = (msg) => app.statuses.push(msg);
    app.resetTTSPlayback = () => {};
    app.resetReadinessPanel = () => {};
    app.updateUIState = (s) => app.uiStates.push(s);
    app._reconnectWebSocket = () => app.reconnects.push(Date.now());
    return app;
}

function fail(msg) {
    console.error('FAIL: ' + msg);
    process.exit(1);
}

// 1) Schedules a reconnect attempt after the configured backoff delay.
const app = makeApp();
app._reconnectMaxAttempts = 2;
app._reconnectBackoffMs = [10, 10];
app._scheduleReconnect();
if (app.wsConnectionState !== 'reconnecting') {
    fail('wsConnectionState should be reconnecting after schedule');
}
if (app.reconnectAttempts !== 1) {
    fail('reconnectAttempts should be 1 after schedule');
}
setTimeout(() => {
    if (app.reconnects.length !== 1) {
        fail('_reconnectWebSocket not called after backoff delay');
    }

    // 2) Exceeding max attempts must stop retrying and surface an error.
    app.reconnects = [];
    app.reconnectAttempts = 2; // == _reconnectMaxAttempts
    app.audioRecorder.isRecording = true;
    app.audioRecorder.isPaused = false;
    app._scheduleReconnect();
    if (app.reconnectAttempts !== 0) {
        fail('reconnectAttempts should reset after giving up');
    }
    if (app.isStreaming !== false) {
        fail('isStreaming should be false after giving up');
    }
    if (app.statuses.length !== 1 || !/Unable to reconnect/.test(app.statuses[0])) {
        fail('expected an "Unable to reconnect" error status');
    }
    if (app.uiStates[app.uiStates.length - 1] !== 'ready') {
        fail('expected UI reset to ready state');
    }
    if (app.audioRecorder.isRecording) {
        fail('recorder should be stopped after giving up');
    }
    setTimeout(() => {
        if (app.reconnects.length !== 0) {
            fail('no reconnect should be scheduled after giving up');
        }
        console.log('PASS');
    }, 50);
}, 60);
"""


def test_schedule_reconnect_constructor_fields_present():
    source = APP_JS.read_text(encoding="utf-8")
    for field in CONSTRUCTOR_FIELDS:
        assert field in source, f"constructor no longer initializes: {field}"


@pytest.mark.skipif(
    not HAS_NODE_APP_SYNTAX, reason="node with ES2020 support not available"
)
def test_schedule_reconnect_end_to_end(tmp_path):
    script = tmp_path / "drive_reconnect.js"
    script.write_text(NODE_DRIVER, encoding="utf-8")
    result = subprocess.run(
        ["node", str(script), str(APP_JS)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "PASS" in result.stdout, result.stdout + result.stderr
