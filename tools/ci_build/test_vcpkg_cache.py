# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/python"))
from util.vcpkg_helpers import generate_windows_triplets  # noqa: E402

ACTION = yaml.safe_load((ROOT / ".github/actions/setup-vcpkg-cache/action.yml").read_text(encoding="utf-8"))
CONFIGURE = next(step for step in ACTION["runs"]["steps"] if step.get("id") == "configure")


class TestVcpkgCache(unittest.TestCase):
    def configure(self, compiler_bytes=b"MSVC patch 1", **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiler = root / "cl.exe"
            if compiler_bytes is not None:
                compiler.write_bytes(compiler_bytes)
            output = root / "output"
            env_file = root / "env"
            env = {
                "RUNNER_OS": "Windows",
                "RUNNER_ARCH": "X64",
                "VCPKG_CACHE_VARIANT": "windows-x64-build-relwithdebinfo",
                "VCPKG_CACHE_PATH": str(root / "binary-cache"),
                "VCToolsVersion": "14.44.35207",
                "WindowsSDKVersion": "10.0.26100.0\\",
                "VSCMD_ARG_HOST_ARCH": "x64",
                "VSCMD_ARG_TGT_ARCH": "x64",
                "GITHUB_OUTPUT": str(output),
                "GITHUB_ENV": str(env_file),
                **overrides,
            }
            with (
                patch.dict(os.environ, env, clear=True),
                patch("shutil.which", return_value=str(compiler) if compiler_bytes is not None else None) as which,
                patch("builtins.print"),
            ):
                exec(compile(CONFIGURE["run"], "configure-vcpkg-cache", "exec"), {})
            prefix = output.read_text(encoding="utf-8").strip().removeprefix("prefix=")
            if env["RUNNER_OS"] == "Windows":
                self.assertEqual(
                    env_file.read_text(encoding="utf-8"), f"VCPKG_DEFAULT_BINARY_CACHE={root / 'binary-cache'}\n"
                )
                self.assertTrue((root / "binary-cache").is_dir())
                which.assert_called_once_with("cl")
            else:
                which.assert_not_called()
                self.assertFalse(env_file.exists())
            return prefix

    def test_selected_compiler_identity_is_stable_across_workspaces(self):
        self.assertEqual(self.configure(), self.configure())
        self.assertEqual(self.configure(WindowsSDKVersion="10.0.26100.0/"), self.configure())

    def test_servicing_update_without_directory_version_change_gets_new_cache(self):
        self.assertNotEqual(self.configure(), self.configure(compiler_bytes=b"MSVC patch 2"))

    def test_sdk_architecture_and_build_variants_remain_separate(self):
        baseline = self.configure()
        for overrides in (
            {"VCToolsVersion": "14.45.35300"},
            {"WindowsSDKVersion": "10.0.26200.0\\"},
            {"VSCMD_ARG_HOST_ARCH": "arm64"},
            {"VSCMD_ARG_TGT_ARCH": "arm64"},
            {"RUNNER_ARCH": "ARM64"},
            {"VCPKG_CACHE_VARIANT": "windows-x64-build-debug"},
        ):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(baseline, self.configure(**overrides))

    def test_missing_selected_toolchain_fails(self):
        for name in ("VCToolsVersion", "WindowsSDKVersion", "VSCMD_ARG_HOST_ARCH", "VSCMD_ARG_TGT_ARCH"):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, name):
                self.configure(**{name: ""})
        with self.assertRaisesRegex(RuntimeError, "Initialize the selected MSVC environment"):
            self.configure(compiler_bytes=None)

    def test_non_windows_cache_identity_is_unchanged(self):
        for runner_os in ("Linux", "macOS"):
            with self.subTest(runner_os=runner_os):
                self.assertEqual(
                    self.configure(RUNNER_OS=runner_os),
                    f"vcpkg-cache-v2-{runner_os}-X64-windows-x64-build-relwithdebinfo-",
                )

    def test_restore_fallback_cannot_cross_toolchains_and_prs_do_not_save(self):
        caches = [step for step in ACTION["runs"]["steps"] if step.get("uses", "").startswith("actions/cache")]
        self.assertEqual(len(caches), 2)
        for cache in caches:
            self.assertTrue(cache["with"]["key"].startswith("${{ steps.configure.outputs.prefix }}"))
            self.assertEqual(cache["with"]["restore-keys"].strip(), "${{ steps.configure.outputs.prefix }}")
            self.assertIn("steps.tool-info.outputs.vcpkg-sha512", cache["with"]["key"])
            self.assertIn("tools/python/util/vcpkg_helpers.py", cache["with"]["key"])
        self.assertEqual(caches[0]["if"], "github.ref == 'refs/heads/main'")
        self.assertTrue(caches[1]["uses"].startswith("actions/cache/restore@"))
        self.assertEqual(caches[1]["if"], "github.ref != 'refs/heads/main'")

    def test_windows_callers_initialize_selected_compiler_before_cache(self):
        callers = 0
        for workflow_path in (ROOT / ".github/workflows").glob("*.yml"):
            workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
            for job in workflow.get("jobs", {}).values():
                steps = job.get("steps", [])
                for index, step in enumerate(steps):
                    if step.get("uses") != "./.github/actions/setup-vcpkg-cache":
                        continue
                    if "windows" not in workflow_path.name:
                        continue
                    with self.subTest(workflow=workflow_path.name):
                        self.assertTrue(
                            any(
                                prior.get("uses") == "./.github/actions/locate-vcvarsall-and-setup-env"
                                for prior in steps[:index]
                            )
                        )
                    callers += 1
        self.assertGreater(callers, 0)


@unittest.skipUnless(os.environ.get("ORT_ACTIONLINT_EXE"), "Set ORT_ACTIONLINT_EXE to validate composite step syntax")
class TestVcpkgCacheWorkflowSyntax(unittest.TestCase):
    def test_composite_steps_in_typed_workflow(self):
        workflow = {
            "name": "Validate vcpkg composite steps",
            "on": {"workflow_dispatch": {"inputs": {name: {"type": "string"} for name in ("key", "path")}}},
            "jobs": {"cache": {"runs-on": "windows-latest", "steps": ACTION["runs"]["steps"]}},
        }
        result = subprocess.run(
            [os.environ["ORT_ACTIONLINT_EXE"], "-shellcheck=", "-pyflakes=", "-"],
            input=yaml.safe_dump(workflow, sort_keys=False),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("ORT_VCPKG_CACHE_TEST_EXE"),
    "Set ORT_VCPKG_CACHE_TEST_EXE to the pinned vcpkg.exe in an initialized MSVC environment",
)
class TestVcpkgBinaryReuse(unittest.TestCase):
    def test_fresh_install_restores_compiled_binary_with_same_toolchain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_windows_triplets(str(root / "triplets"), {"Release"}, "", False)
            port = root / "ports/ort-cache-probe"
            port.mkdir(parents=True)
            (port / "vcpkg.json").write_text('{"name":"ort-cache-probe","version-string":"1"}', encoding="utf-8")
            (port / "probe.c").write_text("int ort_cache_probe(void) { return 42; }\n", encoding="utf-8")
            (port / "probe.h").write_text("int ort_cache_probe(void);\n", encoding="utf-8")
            (port / "copyright").write_text("Test fixture\n", encoding="utf-8")
            (port / "portfile.cmake").write_text(
                """
separate_arguments(flags NATIVE_COMMAND "${VCPKG_C_FLAGS}")
vcpkg_execute_required_process(
    COMMAND cl.exe /nologo /c /MD ${flags} "${CURRENT_PORT_DIR}/probe.c"
        "/Fo${CURRENT_BUILDTREES_DIR}/probe.obj"
    WORKING_DIRECTORY "${CURRENT_BUILDTREES_DIR}" LOGNAME compile)
vcpkg_execute_required_process(
    COMMAND lib.exe /nologo "/OUT:${CURRENT_BUILDTREES_DIR}/probe.lib" "${CURRENT_BUILDTREES_DIR}/probe.obj"
    WORKING_DIRECTORY "${CURRENT_BUILDTREES_DIR}" LOGNAME archive)
file(INSTALL "${CURRENT_BUILDTREES_DIR}/probe.lib" DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
file(INSTALL "${CURRENT_PORT_DIR}/probe.h" DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(INSTALL "${CURRENT_PORT_DIR}/copyright" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
""",
                encoding="utf-8",
            )
            results = []
            prefixes = []
            for name in ("cold", "warm"):
                install_root = root / name
                cache_path = root / f"{name}-cache"
                output = root / f"{name}-output"
                env_file = root / f"{name}-env"
                with patch.dict(
                    os.environ,
                    {
                        "RUNNER_OS": "Windows",
                        "RUNNER_ARCH": "X64",
                        "VCPKG_CACHE_VARIANT": "windows-x64-build-release",
                        "VCPKG_CACHE_PATH": str(cache_path),
                        "GITHUB_OUTPUT": str(output),
                        "GITHUB_ENV": str(env_file),
                    },
                ):
                    exec(compile(CONFIGURE["run"], "configure-vcpkg-cache", "exec"), {})
                prefixes.append(output.read_text(encoding="utf-8"))
                variable, value = env_file.read_text(encoding="utf-8").strip().split("=", 1)
                if name == "warm":
                    self.assertTrue(list((root / "cold-cache").rglob("*.zip")))
                    shutil.copytree(root / "cold-cache", cache_path, dirs_exist_ok=True)
                command = [
                    os.environ["ORT_VCPKG_CACHE_TEST_EXE"],
                    "install",
                    "ort-cache-probe:x64-windows-static-md",
                    "--classic",
                    f"--overlay-ports={root / 'ports'}",
                    f"--overlay-triplets={root / 'triplets/Release/binskim'}",
                    f"--x-install-root={install_root}",
                    f"--x-buildtrees-root={root / 'buildtrees' / name}",
                    f"--x-packages-root={root / 'packages' / name}",
                    "--binarysource=clear",
                    "--binarysource=default,readwrite",
                ]
                result = subprocess.run(
                    command, env={**os.environ, variable: value}, capture_output=True, text=True, check=False
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                results.append(result.stdout + result.stderr)
                self.assertTrue((install_root / "x64-windows-static-md/lib/probe.lib").is_file())
            self.assertEqual(prefixes[0], prefixes[1])
            self.assertEqual(
                (root / "cold/x64-windows-static-md/lib/probe.lib").read_bytes(),
                (root / "warm/x64-windows-static-md/lib/probe.lib").read_bytes(),
            )
            self.assertIn("Restored 0 package(s)", results[0])
            self.assertIn("Building ort-cache-probe:", results[0])
            self.assertIn(Path(shutil.which("cl")).as_posix().lower(), results[0].lower())
            self.assertIn("Restored 1 package(s)", results[1])
            self.assertNotIn("Building ort-cache-probe:", results[1])
            print("vcpkg binary reuse: cold built 1 package; fresh warm install restored 1 package without rebuilding.")


if __name__ == "__main__":
    unittest.main()
