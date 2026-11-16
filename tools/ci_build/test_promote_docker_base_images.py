# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Offline tests: no registry, Azure login, Docker daemon or publishing is used."""

import base64
import copy
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

with mock.patch.object(sys, "path", [str(Path(__file__).resolve().parent), *sys.path]):
    import promote_docker_base_images as promotion
    import resolve_docker_base_image as resolver

REPO_ROOT = Path(__file__).resolve().parents[2]
TRUSTED_CONTEXT = {
    "GITHUB_REPOSITORY": "microsoft/onnxruntime",
    "GITHUB_EVENT_NAME": "workflow_dispatch",
    "GITHUB_REF": "refs/heads/main",
    "GITHUB_REF_PROTECTED": "true",
    "GITHUB_WORKFLOW_REF": ("microsoft/onnxruntime/.github/workflows/promote_base_images.yml@refs/heads/main"),
}


def image_fixture():
    # Different platforms and an unmodified index are deliberately not the host architecture.
    raw = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": promotion.MANIFEST_TYPES[0],
            "manifests": [
                {"digest": "sha256:" + "1" * 64, "platform": {"os": "linux", "architecture": arch}}
                for arch in ("arm64", "amd64")
            ],
        }
    ).encode()
    image = {
        "repository": resolver.ACR_BASE_PREFIX + "cpu_test",
        "tag": "20261105.1",
        "digest": promotion.digest_bytes(raw),
        "producer_build": "123",
    }
    return raw, image


def http_error(status, body=b""):
    return urllib.error.HTTPError("https://ghcr.io", status, "test", {}, io.BytesIO(body))


class InventoryTests(unittest.TestCase):
    def test_inventory_selection_and_default_github_mapping(self):
        manifest = resolver.load_manifest()
        for family, image in manifest["images"].items():
            with self.subTest(family=family):
                self.assertEqual(promotion.approved_images(family), {family: image})
                self.assertEqual(
                    resolver.resolve_family(family, registry="ghcr"),
                    f"{resolver.GHCR_BASE_PREFIX}{family}@{image['digest']}",
                )
                self.assertEqual(resolver.resolve_family(family), f"{image['repository']}@{image['digest']}")
        for dockerfile, family in manifest["dockerfile_defaults"].items():
            arguments = resolver.resolve_build_args(resolver.DOCKER_DIR / dockerfile, registry="ghcr")
            self.assertEqual(
                arguments, ["--build-arg", f"BASEIMAGE={resolver.resolve_family(family, registry='ghcr')}"]
            )

    def test_rejects_unapproved_selection_before_credentials(self):
        for selection in ("unknown", "../cpu_test", "all; echo unsafe", "--all", ""):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                promotion.approved_images(selection)

    def test_rejects_invalid_pins_namespaces_and_provenance(self):
        manifest = resolver.load_manifest()
        family = next(iter(manifest["images"]))
        for field, value in (
            ("digest", "sha256:abc"),
            ("digest", "sha256:" + "A" * 64),
            ("tag", "../latest"),
            ("repository", "untrusted.example/test"),
            ("repository", resolver.ACR_BASE_PREFIX + "different_family"),
            ("producer_build", "unknown"),
        ):
            changed = copy.deepcopy(manifest)
            changed["images"][family][field] = value
            with (
                self.subTest(field=field, value=value),
                tempfile.TemporaryDirectory() as directory,
                mock.patch.object(resolver, "MANIFEST_PATH", Path(directory) / "inventory.json"),
            ):
                resolver.MANIFEST_PATH.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    promotion.approved_images(family)

    def test_explicit_overrides_and_ado_behavior_are_preserved(self):
        arguments = "--build-arg BASEIMAGE=custom.example/image@sha256:abc --build-arg FOO=bar"
        self.assertEqual(
            resolver.resolve_build_args("custom/Dockerfile", arguments, registry="ghcr"), shlex.split(arguments)
        )
        with self.assertRaisesRegex(ValueError, "either"):
            resolver.resolve_build_args("custom/Dockerfile", arguments, "cpu_test", "ghcr")


class GhcrTests(unittest.TestCase):
    def request_failure(self, error, **kwargs):
        opener = mock.Mock()
        opener.open.side_effect = error
        with mock.patch.object(promotion.urllib.request, "build_opener", return_value=opener):
            return promotion.request_bytes("https://ghcr.io/v2/test/manifests/test", {}, **kwargs)

    def test_only_structured_manifest_absence_returns_none(self):
        for code in ("MANIFEST_UNKNOWN", "NAME_UNKNOWN"):
            with self.subTest(code=code):
                body = json.dumps({"errors": [{"code": code}]}).encode()
                self.assertIsNone(self.request_failure(http_error(404, body), allow_missing=True))

    def test_nonabsence_errors_never_recommend_promotion(self):
        for error, expected in (
            (http_error(401), "authentication/access"),
            (http_error(403), "authentication/access"),
            (http_error(429), "rate limit"),
            (http_error(500), "HTTP 500"),
            (http_error(404, b"not json"), "unconfirmed"),
            (http_error(404, b'{"errors":[{"code":"DENIED"}]}'), "unconfirmed"),
            (urllib.error.URLError("TLS failed"), "network/TLS"),
            (TimeoutError(), "network/TLS"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(promotion.RegistryError, expected) as caught:
                self.request_failure(error, allow_missing=True)
            self.assertNotIn("gh workflow run", str(caught.exception))

    def test_token_endpoint_404_is_not_image_absence(self):
        with self.assertRaisesRegex(promotion.RegistryError, "unconfirmed"):
            self.request_failure(http_error(404, b'{"errors":[{"code":"NAME_UNKNOWN"}]}'))

    def test_anonymous_and_publisher_auth_are_separate_and_constrained(self):
        for username, password, access in (("", "", "pull"), ("actor", "secret", "pull%2Cpush")):
            with self.subTest(access=access), mock.patch.object(promotion, "request_bytes") as request:
                request.side_effect = [b'{"token":"test-token"}', b"manifest"]
                client = promotion.Ghcr(username, password)
                self.assertEqual(client.manifest(resolver.GHCR_BASE_PREFIX + "cpu_test", "tag"), b"manifest")
                token_call, manifest_call = request.call_args_list
                self.assertIn(f"%3A{access}", token_call.args[0])
                self.assertEqual(bool(token_call.args[1]), bool(password))
                self.assertEqual(manifest_call.args[1]["Authorization"], "Bearer test-token")
                self.assertIn(promotion.MANIFEST_TYPES[0], manifest_call.args[1]["Accept"])
                with self.assertRaises(ValueError):
                    client.manifest("ghcr.io/untrusted/image", "tag")
                with self.assertRaises(ValueError):
                    client.manifest(resolver.GHCR_BASE_PREFIX + "../untrusted", "tag")

    def test_tokens_are_refreshed_after_potentially_long_copies(self):
        with mock.patch.object(promotion, "request_bytes") as request:
            request.side_effect = [b'{"token":"first"}', b"manifest", b'{"token":"second"}', b"manifest"]
            client = promotion.Ghcr("actor", "secret")
            for _ in range(2):
                client.manifest(resolver.GHCR_BASE_PREFIX + "cpu_test", "tag")
            self.assertEqual(request.call_args_list[1].args[1]["Authorization"], "Bearer first")
            self.assertEqual(request.call_args_list[3].args[1]["Authorization"], "Bearer second")

    def test_invalid_token_responses_fail_without_logging_response(self):
        for raw in (b"secret-not-json", b'["secret"]', b'{"token":""}', b'{"token":"secret\\nvalue"}'):
            with (
                self.subTest(raw=raw),
                mock.patch.object(promotion, "request_bytes", return_value=raw) as request,
                self.assertRaisesRegex(promotion.RegistryError, "invalid") as caught,
            ):
                promotion.Ghcr().manifest(resolver.GHCR_BASE_PREFIX + "cpu_test", "tag")
            self.assertNotIn("secret", str(caught.exception))
            self.assertEqual(request.call_count, 1)

    def test_redirect_is_rejected_without_forwarding_credentials(self):
        with self.assertRaisesRegex(promotion.RegistryError, "redirect"):
            promotion.NoRedirect().redirect_request(None, None, 302, "", {}, "https://untrusted.example")


class ConsumerTests(unittest.TestCase):
    def setUp(self):
        self.family, self.image = next(iter(resolver.load_manifest()["images"].items()))
        self.reference = f"{resolver.GHCR_BASE_PREFIX}{self.family}@{self.image['digest']}"

    def test_missing_pin_includes_actual_dispatch_inputs_and_no_fallback(self):
        with (
            mock.patch.object(promotion.Ghcr, "manifest", return_value=None),
            self.assertRaises(promotion.RegistryError) as caught,
        ):
            promotion.check_build_args(f"--build-arg BASEIMAGE={self.reference}")
        error = str(caught.exception)
        self.assertIn(f"gh workflow run {promotion.WORKFLOW}", error)
        self.assertIn(f"--ref main -f family={self.family}", error)
        self.assertIn("retry CI", error)
        self.assertIn("will not fall back", error)

    def test_access_error_propagates_without_dispatch_advice(self):
        with (
            mock.patch.object(promotion.Ghcr, "manifest", side_effect=promotion.RegistryError("access denied")),
            self.assertRaisesRegex(promotion.RegistryError, "^access denied$"),
        ):
            promotion.check_build_args(f"--build-arg=BASEIMAGE={self.reference}")

    def test_unapproved_ghcr_digest_fails_before_network(self):
        with mock.patch.object(promotion.Ghcr, "manifest") as manifest:
            for reference in (
                f"{resolver.GHCR_BASE_PREFIX}{self.family}:latest",
                f"{resolver.GHCR_BASE_PREFIX}{self.family}@sha256:{'0' * 64}",
                f"{resolver.GHCR_BASE_PREFIX}unknown@{self.image['digest']}",
            ):
                with self.subTest(reference=reference), self.assertRaises(ValueError):
                    promotion.check_build_args(f"--build-arg BASEIMAGE={reference}")
            manifest.assert_not_called()

    def test_correct_manifest_is_required_even_on_success_response(self):
        raw, image = image_fixture()
        reference = f"{resolver.GHCR_BASE_PREFIX}cpu_test@{image['digest']}"
        with (
            mock.patch.object(promotion, "approved_images", return_value={"cpu_test": image}),
            mock.patch.object(promotion.Ghcr, "manifest", return_value=raw) as manifest,
        ):
            promotion.check_build_args(f"--build-arg BASEIMAGE={reference}")
            manifest.return_value = b"different manifest"
            with self.assertRaisesRegex(promotion.RegistryError, "approved digest"):
                promotion.check_build_args(f"--build-arg BASEIMAGE={reference}")

    def test_custom_non_ghcr_override_does_not_trigger_registry_check(self):
        with mock.patch.object(promotion.Ghcr, "manifest") as manifest:
            promotion.check_build_args("--build-arg BASEIMAGE=custom.example/image:tag")
            manifest.assert_not_called()


class PromotionTests(unittest.TestCase):
    def test_context_rejects_forks_branches_prs_and_unprotected_refs(self):
        for key, value in (
            ("GITHUB_REPOSITORY", "fork/onnxruntime"),
            ("GITHUB_EVENT_NAME", "pull_request"),
            ("GITHUB_EVENT_NAME", "pull_request_target"),
            ("GITHUB_REF", "refs/heads/feature"),
            ("GITHUB_REF", "refs/tags/main"),
            ("GITHUB_REF_PROTECTED", "false"),
            (
                "GITHUB_WORKFLOW_REF",
                TRUSTED_CONTEXT["GITHUB_WORKFLOW_REF"].replace("@refs/heads/main", "@refs/heads/dev"),
            ),
        ):
            context = TRUSTED_CONTEXT | {key: value}
            with (
                self.subTest(key=key, value=value),
                mock.patch.dict(os.environ, context, clear=True),
                mock.patch.object(promotion, "run_command") as command,
                self.assertRaisesRegex(ValueError, "protected main"),
            ):
                promotion.promote("all")
            command.assert_not_called()
        with mock.patch.dict(os.environ, TRUSTED_CONTEXT, clear=True):
            promotion.require_trusted_context("all")

    def test_exact_protected_upstream_pilot_push_is_allowed(self):
        context = TRUSTED_CONTEXT | {
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": f"refs/heads/{promotion.PILOT_BRANCH}",
            "GITHUB_WORKFLOW_REF": (
                f"microsoft/onnxruntime/.github/workflows/{promotion.WORKFLOW}@refs/heads/{promotion.PILOT_BRANCH}"
            ),
        }
        with mock.patch.dict(os.environ, context, clear=True):
            promotion.require_trusted_context("all")

    def test_pilot_rejects_forks_other_events_refs_and_selections_before_credentials(self):
        context = TRUSTED_CONTEXT | {
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": f"refs/heads/{promotion.PILOT_BRANCH}",
            "GITHUB_WORKFLOW_REF": (
                f"microsoft/onnxruntime/.github/workflows/{promotion.WORKFLOW}@refs/heads/{promotion.PILOT_BRANCH}"
            ),
        }
        for key, value in (
            ("GITHUB_REPOSITORY", "fork/onnxruntime"),
            ("GITHUB_EVENT_NAME", "pull_request"),
            ("GITHUB_EVENT_NAME", "pull_request_target"),
            ("GITHUB_EVENT_NAME", "workflow_dispatch"),
            ("GITHUB_REF", "refs/heads/main"),
            ("GITHUB_REF", "refs/heads/another-protected-branch"),
            ("GITHUB_REF", f"refs/tags/{promotion.PILOT_BRANCH}"),
            ("GITHUB_REF_PROTECTED", "false"),
            ("GITHUB_WORKFLOW_REF", TRUSTED_CONTEXT["GITHUB_WORKFLOW_REF"]),
        ):
            with (
                self.subTest(key=key, value=value),
                mock.patch.dict(os.environ, context | {key: value}, clear=True),
                mock.patch.object(promotion, "run_command") as command,
                self.assertRaisesRegex(ValueError, "exact-branch pilot"),
            ):
                promotion.promote("all")
            command.assert_not_called()
        with (
            mock.patch.dict(os.environ, context, clear=True),
            mock.patch.object(promotion, "run_command") as command,
            self.assertRaisesRegex(ValueError, "all approved families"),
        ):
            promotion.promote("cpu_x64_almalinux8_gcc14")
        command.assert_not_called()

    def test_cli_rejects_untrusted_context_before_credentials_with_no_stdout(self):
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "ci_build" / "promote_docker_base_images.py"),
                "promote",
                "--family",
                "all",
            ],
            env=os.environ | {"GITHUB_EVENT_NAME": "pull_request"},
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("protected main", result.stderr)

    def test_all_platforms_and_exact_digest_are_copied_and_verified(self):
        raw, image = image_fixture()
        for existing in (None, raw):
            with (
                self.subTest(existing=existing),
                mock.patch.object(promotion, "run_command", side_effect=[raw, b""]) as command,
            ):
                registry = mock.Mock()
                registry.manifest.side_effect = [existing, existing, raw, raw]
                promotion.copy_image("cpu_test", image, registry, Path("isolated-auth.json"))
                copy_command = command.call_args_list[1].args[0]
                self.assertEqual(
                    copy_command,
                    [
                        "skopeo",
                        "copy",
                        "--all",
                        "--preserve-digests",
                        "--authfile",
                        "isolated-auth.json",
                        f"docker://{image['repository']}@{image['digest']}",
                        f"docker://{resolver.GHCR_BASE_PREFIX}cpu_test:{image['tag']}",
                    ],
                )
                self.assertEqual(
                    [call.args[1] for call in registry.manifest.call_args_list],
                    [image["tag"], image["digest"], image["tag"], image["digest"]],
                )

    def test_conflicts_and_incompatible_content_are_rejected_before_copy(self):
        raw, image = image_fixture()
        for responses, expected in (([b"conflicting tag"], "Conflicting"), ([None, b"wrong content"], "Incompatible")):
            registry = mock.Mock()
            registry.manifest.side_effect = responses
            with (
                self.subTest(expected=expected),
                mock.patch.object(promotion, "run_command", return_value=raw) as command,
                self.assertRaisesRegex(promotion.RegistryError, expected),
            ):
                promotion.copy_image("cpu_test", image, registry, Path("auth.json"))
            self.assertEqual(command.call_count, 1)

    def test_source_mismatch_fails_before_destination_access(self):
        _, image = image_fixture()
        registry = mock.Mock()
        with (
            mock.patch.object(promotion, "run_command", return_value=b"changed source"),
            self.assertRaisesRegex(promotion.RegistryError, "ACR source"),
        ):
            promotion.copy_image("cpu_test", image, registry, Path("auth.json"))
        registry.manifest.assert_not_called()

    def test_copy_failure_and_postcopy_digest_failure_propagate(self):
        raw, image = image_fixture()
        registry = mock.Mock()
        registry.manifest.return_value = None
        with (
            mock.patch.object(promotion, "run_command", side_effect=[raw, RuntimeError("copy failed")]),
            self.assertRaisesRegex(RuntimeError, "copy failed"),
        ):
            promotion.copy_image("cpu_test", image, registry, Path("auth.json"))
        for result in (None, b"altered index"):
            registry.manifest.side_effect = [None, None, raw, result]
            with (
                self.subTest(result=result),
                mock.patch.object(promotion, "run_command", side_effect=[raw, b""]),
                self.assertRaisesRegex(promotion.RegistryError, "Destination digest"),
            ):
                promotion.copy_image("cpu_test", image, registry, Path("auth.json"))

    def test_credentials_are_isolated_and_cleaned_on_success_and_failure(self):
        _, image = image_fixture()
        for failure in (None, RuntimeError, KeyboardInterrupt):
            with tempfile.TemporaryDirectory() as cache:
                context = TRUSTED_CONTEXT | {
                    "GHCR_TOKEN": "fake-github-token",
                    "GITHUB_ACTOR": "actor",
                    "AZURE_CONFIG_DIR": cache,
                }
                authfiles = []

                def copy_image(
                    family, selected, registry, authfile, *, authfiles=authfiles, failure=failure, cache=cache
                ):
                    authfiles.append(authfile)
                    credentials = json.loads(authfile.read_text(encoding="utf-8"))["auths"]
                    self.assertEqual(
                        base64.b64decode(credentials["ghcr.io"]["auth"]).decode(), "actor:fake-github-token"
                    )
                    self.assertEqual(
                        base64.b64decode(credentials[promotion.ACR_REGISTRY]["auth"]).decode(),
                        "00000000-0000-0000-0000-000000000000:fake-acr-token",
                    )
                    self.assertNotIn("GHCR_TOKEN", os.environ)
                    self.assertEqual(authfile.parent.parent, Path(cache))
                    if os.name != "nt":
                        self.assertEqual(authfile.stat().st_mode & 0o777, 0o600)
                        self.assertEqual(authfile.parent.stat().st_mode & 0o777, 0o700)
                    if failure:
                        raise failure("copy failed")

                with (
                    self.subTest(failure=failure),
                    mock.patch.dict(os.environ, context, clear=True),
                    mock.patch.object(promotion, "approved_images", return_value={"cpu_test": image}),
                    mock.patch.object(promotion, "run_command", return_value=b"fake-acr-token"),
                    mock.patch.object(promotion, "copy_image", side_effect=copy_image),
                ):
                    if failure:
                        with self.assertRaisesRegex(failure, "copy failed"):
                            promotion.promote("cpu_test")
                    else:
                        promotion.promote("cpu_test")
                self.assertTrue(authfiles)
                self.assertFalse(authfiles[0].exists())
                self.assertEqual(list(Path(cache).iterdir()), [])

    def test_credential_acquisition_failure_is_cleaned_and_not_logged(self):
        with (
            tempfile.TemporaryDirectory() as cache,
            mock.patch.dict(
                os.environ,
                TRUSTED_CONTEXT | {"GHCR_TOKEN": "fake-token", "GITHUB_ACTOR": "actor", "AZURE_CONFIG_DIR": cache},
                clear=True,
            ),
            mock.patch.object(
                promotion.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(["az"], 1, b"secret stdout", b"secret stderr"),
            ),
            self.assertRaisesRegex(RuntimeError, "az acr failed") as caught,
        ):
            promotion.promote("all")
        self.assertNotIn("secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
