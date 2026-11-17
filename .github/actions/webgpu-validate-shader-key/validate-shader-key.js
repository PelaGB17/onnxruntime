// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

"use strict";

const { spawnSync } = require("child_process");
const fs = require("fs");
const { EOL, tmpdir } = require("os");
const path = require("path");
const readline = require("readline");

// This script is used to parse the raw log data and check if there are inconsistent shader.
// Within the same device configuration, the same shader key must produce the same code.

const shaderMap = new Map();

const regexShaderStart =
  /^===\ WebGPU\ Shader\ code\ \[.+?, Configuration="(?<configuration>[^"]+)", Key="(?<key>.+)"]\ Start\ ===$/;
const regexShaderEnd =
  /^===\ WebGPU\ Shader\ code\ \[.+?, Configuration="(?<configuration>[^"]+)", Key="(?<key>.+)"]\ End\ ===$/;

function getShaderDiff(previousCode, currentCode) {
  let directory;
  try {
    directory = fs.mkdtempSync(path.join(tmpdir(), "webgpu-shader-diff-"));
    fs.writeFileSync(path.join(directory, "previous.wgsl"), previousCode);
    fs.writeFileSync(path.join(directory, "current.wgsl"), currentCode);
    const result = spawnSync(
      "git",
      [
        "--no-pager",
        "diff",
        "--no-index",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--text",
        "--",
        "previous.wgsl",
        "current.wgsl",
      ],
      { cwd: directory, encoding: "utf8", windowsHide: true }
    );
    return !result.error && result.status === 1 && result.stdout
      ? result.stdout.trimEnd()
      : null;
  } catch {
    return null;
  } finally {
    if (directory) {
      try {
        fs.rmSync(directory, { recursive: true, force: true });
      } catch (error) {
        console.error(`Failed to remove shader diff directory "${directory}": ${error.message}`);
      }
    }
  }
}

async function processVerboseLog() {
  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });

  let currentShaderKey = null;
  let currentShaderConfiguration = null;
  let currentShaderCode = null;
  let currentShaderStartLineNumber = null;
  let lineNumber = 0;
  let inconsistentShaderCount = 0;

  for await (const line of rl) {
    lineNumber++;
    const resultStart = regexShaderStart.exec(line);
    if (resultStart) {
      if (currentShaderKey) {
        throw new Error(
          `Line ${lineNumber}: Found incomplete shader code for key "${currentShaderKey}" (shader started at line ${currentShaderStartLineNumber}).`
        );
      }

      currentShaderKey = resultStart.groups.key;
      currentShaderConfiguration = resultStart.groups.configuration;
      currentShaderCode = "";
      currentShaderStartLineNumber = lineNumber;
      continue;
    }

    const resultEnd = regexShaderEnd.exec(line);
    if (resultEnd) {
      if (!currentShaderKey) {
        throw new Error(
          `Line ${lineNumber}: Found unexpected shader end for key "${resultEnd.groups.key}".`
        );
      }

      if (
        resultEnd.groups.key !== currentShaderKey ||
        resultEnd.groups.configuration !== currentShaderConfiguration
      ) {
        throw new Error(
          `Line ${lineNumber}: Shader end key or configuration does not match its start at line ${currentShaderStartLineNumber}.`
        );
      }

      const comparisonKey = JSON.stringify([
        currentShaderConfiguration,
        currentShaderKey,
      ]);
      if (shaderMap.has(comparisonKey)) {
        const previousShader = shaderMap.get(comparisonKey);
        if (previousShader.code !== currentShaderCode) {
          inconsistentShaderCount++;
          const diff = getShaderDiff(previousShader.code, currentShaderCode);
          const details = diff ? `=== Shader Diff ===
${diff}` : `=== Previous Shader Start ===
${previousShader.code}
=== Previous Shader End ===

=== Current Shader Start ===
${currentShaderCode}
=== Current Shader End ===`;
          console.error(`Line ${lineNumber}: Found inconsistent shader code for key "${currentShaderKey}" (configuration ${JSON.stringify(currentShaderConfiguration)}, previous shader started at line ${previousShader.startLineNumber}, current shader started at line ${currentShaderStartLineNumber}).
${details}`);
        }
      } else {
        shaderMap.set(comparisonKey, {
          code: currentShaderCode,
          startLineNumber: currentShaderStartLineNumber,
        });
      }

      currentShaderKey = null;
      currentShaderConfiguration = null;
      currentShaderCode = null;
      currentShaderStartLineNumber = null;
      continue;
    }

    if (line.startsWith("=== WebGPU Shader code [")) {
      throw new Error(
        `Line ${lineNumber}: Shader header must include nonempty Configuration and Key fields.`
      );
    }

    if (currentShaderKey) {
      currentShaderCode += line + EOL;
    }
  }

  if (currentShaderKey) {
    throw new Error(
      `Reached end of log (line ${lineNumber}) with incomplete shader code for key "${currentShaderKey}" (shader started at line ${currentShaderStartLineNumber}).`
    );
  }

  if (shaderMap.size === 0) {
    throw new Error("No shader code found.");
  }

  if (inconsistentShaderCount > 0) {
    throw new Error(
      `Found ${inconsistentShaderCount} inconsistent shader code occurrences across ${shaderMap.size} shader keys.`
    );
  }

  console.log(
    `All shader code is consistent. Total ${shaderMap.size} shader keys found.`
  );
}

processVerboseLog();
