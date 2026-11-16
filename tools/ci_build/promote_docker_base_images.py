#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Promote approved ACR bases, or anonymously check pinned GHCR bases before CI."""

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from resolve_docker_base_image import GHCR_BASE_PREFIX, ghcr_repository, load_manifest

WORKFLOW = "promote_base_images.yml"
PILOT_BRANCH = "eserscor-2026-11-12-base-image-promotion"
ACR_REGISTRY = "onnxruntimebuildcache.azurecr.io"
MANIFEST_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
)


class RegistryError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RegistryError("Unexpected registry redirect; refusing to forward credentials")


def request_bytes(url, headers, *, allow_missing=False):
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        # A bare 404 (or denied token exchange) does not establish image absence.
        if allow_missing and error.code == 404:
            try:
                body = json.loads(error.read())
                codes = {entry["code"] for entry in body["errors"]}
            except (ValueError, KeyError, TypeError) as parse_error:
                raise RegistryError("Unrecognized GHCR 404 response; image absence is unconfirmed") from parse_error
            if codes and codes <= {"MANIFEST_UNKNOWN", "NAME_UNKNOWN"}:
                return None
        if error.code in (401, 403):
            raise RegistryError(
                "GHCR authentication/access failed; absence is unconfirmed. "
                "Check package visibility and workflow access; do not assume promotion is needed."
            ) from None
        if error.code == 429:
            raise RegistryError("GHCR rate limit exceeded; retry after the registry limit resets") from None
        raise RegistryError(f"GHCR request failed with HTTP {error.code}; absence is unconfirmed") from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RegistryError("GHCR network/TLS request failed; absence is unconfirmed") from error


class Ghcr:
    def __init__(self, username="", password=""):
        self.authorization = {}
        self.access = "pull"
        if password:
            encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
            self.authorization = {"Authorization": f"Basic {encoded}"}
            self.access = "pull,push"

    def manifest(self, repository, reference):
        if not re.fullmatch(re.escape(GHCR_BASE_PREFIX) + r"[a-z0-9]+(?:[._-][a-z0-9]+)*", repository):
            raise ValueError("GHCR repository is outside the approved base namespace")
        if not re.fullmatch(r"(?:sha256:[0-9a-f]{64}|[\w][\w.-]{0,127})", reference, flags=re.ASCII):
            raise ValueError("Invalid manifest reference")
        name = repository.removeprefix("ghcr.io/")
        # Registry bearer tokens can expire while a large multi-platform copy runs.
        query = urllib.parse.urlencode({"service": "ghcr.io", "scope": f"repository:{name}:{self.access}"})
        raw_token = request_bytes(f"https://ghcr.io/token?{query}", self.authorization)
        try:
            token_response = json.loads(raw_token)
        except ValueError as error:
            raise RegistryError("GHCR returned an invalid token response") from error
        if not isinstance(token_response, dict):
            raise RegistryError("GHCR returned an invalid token response")
        token = token_response.get("token") or token_response.get("access_token")
        if not isinstance(token, str) or not token or "\n" in token or "\r" in token:
            raise RegistryError("GHCR returned an invalid registry token")
        return request_bytes(
            f"https://ghcr.io/v2/{name}/manifests/{reference}",
            {"Authorization": f"Bearer {token}", "Accept": ", ".join(MANIFEST_TYPES)},
            allow_missing=True,
        )


def digest_bytes(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def approved_images(selection):
    manifest = load_manifest()
    images = manifest["images"]
    if selection != "all" and selection not in images:
        raise ValueError(f"Unknown approved base-image family: {selection!r}")
    selected = images if selection == "all" else {selection: images[selection]}
    for family, image in selected.items():
        ghcr_repository(family, image)
        producer = manifest["producer_builds"].get(image["producer_build"], {})
        if not re.fullmatch(r"[0-9a-f]{40}", producer.get("source_commit", "")):
            raise ValueError(f"Missing approved producer metadata for {family!r}")
    return selected


def check_build_args(build_args):
    images = approved_images("all")
    approved = {f"{ghcr_repository(family, image)}@{image['digest']}": family for family, image in images.items()}
    for argument in shlex.split(build_args):
        if not argument.startswith(("BASEIMAGE=", "--build-arg=BASEIMAGE=")):
            continue
        reference = argument.split("BASEIMAGE=", 1)[1]
        if not reference.startswith(GHCR_BASE_PREFIX):
            continue
        if reference not in approved:
            raise ValueError(f"GHCR base is not an approved inventory digest: {reference}")
        repository, digest = reference.split("@")
        raw = Ghcr().manifest(repository, digest)
        if raw is None:
            family = approved[reference]
            raise RegistryError(
                f"Required GHCR base image is missing: {reference}. An authorized upstream maintainer must "
                f"complete redistribution/public-visibility approvals, then run 'Promote approved base images' "
                f"({WORKFLOW}) on main with family={family}, and retry CI. For example: "
                f"gh workflow run {WORKFLOW} --repo microsoft/onnxruntime --ref main -f family={family}. "
                f"Before the workflow reaches main, an authorized maintainer can instead rerun the approved "
                f"upstream promotion run for {PILOT_BRANCH} in Actions; that pilot promotes all approved families. "
                "This job will not fall back to ACR or trigger publication."
            )
        if digest_bytes(raw) != digest:
            raise RegistryError(f"GHCR manifest does not match the approved digest for {approved[reference]}")


def require_trusted_context(selection):
    event = os.environ.get("GITHUB_EVENT_NAME")
    ref = os.environ.get("GITHUB_REF")
    allowed = (event == "workflow_dispatch" and ref == "refs/heads/main") or (
        event == "push" and ref == f"refs/heads/{PILOT_BRANCH}" and selection == "all"
    )
    expected = {
        "GITHUB_REPOSITORY": "microsoft/onnxruntime",
        "GITHUB_REF_PROTECTED": "true",
        "GITHUB_WORKFLOW_REF": f"microsoft/onnxruntime/.github/workflows/{WORKFLOW}@{ref}",
    }
    if not allowed or any(os.environ.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Promotion requires the upstream workflow_dispatch workflow on protected main "
            "or a protected, exact-branch pilot push promoting all approved families"
        )


def run_command(arguments):
    result = subprocess.run(arguments, capture_output=True, check=False)
    if result.returncode:
        # Never echo credential-bearing command output (including Azure CLI errors).
        raise RuntimeError(f"{arguments[0]} {arguments[1]} failed (exit {result.returncode}); no fallback performed")
    return result.stdout


def copy_image(family, image, registry, authfile):
    repository = ghcr_repository(family, image)
    digest, tag = image["digest"], image["tag"]
    source = f"{image['repository']}@{digest}"
    source_raw = run_command(["skopeo", "inspect", "--authfile", str(authfile), "--raw", f"docker://{source}"])
    if digest_bytes(source_raw) != digest:
        raise RegistryError(f"ACR source manifest differs from the approved digest for {family}")
    existing = registry.manifest(repository, tag)
    if existing is not None and digest_bytes(existing) != digest:
        raise RegistryError(f"Conflicting destination tag {repository}:{tag}; refusing to overwrite")
    pinned = registry.manifest(repository, digest)
    if pinned is not None and pinned != source_raw:
        raise RegistryError(f"Incompatible existing destination content for {family}")
    # Do not skip the full-platform copy based only on the root manifest's presence.
    run_command(
        [
            "skopeo",
            "copy",
            "--all",
            "--preserve-digests",
            "--authfile",
            str(authfile),
            f"docker://{source}",
            f"docker://{repository}:{tag}",
        ]
    )
    for reference in (tag, digest):
        raw = registry.manifest(repository, reference)
        if raw is None or raw != source_raw or digest_bytes(raw) != digest:
            raise RegistryError(f"Destination digest verification failed for {repository}:{reference}")
    print(f"Verified {repository}:{tag} -> {digest}")


def promote(selection):
    require_trusted_context(selection)
    images = approved_images(selection)
    password = os.environ.pop("GHCR_TOKEN", "")
    username = os.environ.get("GITHUB_ACTOR", "")
    if not password or not username:
        raise ValueError("Promotion requires GITHUB_ACTOR and the workflow's GHCR_TOKEN")
    with tempfile.TemporaryDirectory(prefix="ort-base-promotion-", dir=os.environ["AZURE_CONFIG_DIR"]) as directory:
        root = Path(directory)
        root.chmod(0o700)
        acr_token = (
            run_command(
                [
                    "az",
                    "acr",
                    "login",
                    "--name",
                    "onnxruntimebuildcache",
                    "--expose-token",
                    "--query",
                    "accessToken",
                    "--output",
                    "tsv",
                    "--only-show-errors",
                ]
            )
            .decode()
            .strip()
        )
        if not acr_token or "\n" in acr_token or "\r" in acr_token:
            raise ValueError("Azure CLI did not return a valid ACR read token")
        authfile = root / "auth.json"
        credentials = {
            ACR_REGISTRY: ("00000000-0000-0000-0000-000000000000", acr_token),
            "ghcr.io": (username, password),
        }
        with authfile.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "auths": {
                        host: {"auth": base64.b64encode(f"{user}:{token}".encode()).decode()}
                        for host, (user, token) in credentials.items()
                    }
                },
                stream,
            )
        authfile.chmod(0o600)
        registry = Ghcr(username, password)
        for family, image in images.items():
            copy_image(family, image, registry, authfile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "promote"):
        commands.add_parser(command).add_argument("--family", required=True)
    commands.add_parser("check").add_argument("--build-args", required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            check_build_args(args.build_args)
        elif args.command == "validate":
            require_trusted_context(args.family)
            approved_images(args.family)
        else:
            promote(args.family)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"Base-image {args.command} failed: {error}\n")


if __name__ == "__main__":
    main()
