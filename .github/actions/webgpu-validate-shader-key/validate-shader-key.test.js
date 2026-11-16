// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

"use strict";

const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const path = require("node:path");
const { test } = require("node:test");

function shader(key, code, configuration = "test-device") {
  const header = `=== WebGPU Shader code [test, Configuration="${configuration}", Key="${key}"]`;
  return `${header} Start ===\n${code}\n${header} End ===\n`;
}

function validate(input) {
  const result = spawnSync(
    process.execPath,
    [path.join(__dirname, "validate-shader-key.js")],
    { input, encoding: "utf8" }
  );
  assert.ifError(result.error);
  assert.equal(result.signal, null);
  return result;
}

test("reports every collision before failing, including later variants of the same key", () => {
  const result = validate(
    shader("first", "original first") +
      shader("second", "original second") +
      shader("first", "changed first") +
      shader("second", "changed second") +
      shader("first", "another first")
  );

  assert.notEqual(result.status, 0);
  assert.equal(
    (result.stderr.match(/Found inconsistent shader code for key/g) || []).length,
    3
  );
  assert.match(result.stderr, /key "first".*shader started at line 7/);
  assert.match(result.stderr, /key "second".*shader started at line 10/);
  assert.match(result.stderr, /key "first".*shader started at line 13/);
  for (const code of [
    "original first",
    "original second",
    "changed first",
    "changed second",
    "another first",
  ]) {
    assert.ok(result.stderr.includes(code));
  }
  assert.match(
    result.stderr,
    /Found 3 inconsistent shader code occurrences across 2 shader keys/
  );
  assert.doesNotMatch(result.stdout, /All shader code is consistent/);
});

test("accepts matching shaders and compares configurations separately", () => {
  const result = validate(
    shader("shared", "first code") +
      shader("shared", "first code") +
      shader("shared", "second code", "other-device")
  );

  assert.equal(result.status, 0);
  assert.equal(result.stderr, "");
  assert.match(
    result.stdout,
    /All shader code is consistent. Total 2 shader keys found/
  );
});

test("still rejects incomplete shaders after reporting collisions", () => {
  const result = validate(
    shader("shared", "original") +
      shader("shared", "changed") +
      '=== WebGPU Shader code [test, Configuration="test-device", Key="unfinished"] Start ===\n'
  );

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Found inconsistent shader code for key "shared"/);
  assert.match(result.stderr, /Reached end of log .* incomplete shader code/);
  assert.doesNotMatch(result.stdout, /All shader code is consistent/);
});
