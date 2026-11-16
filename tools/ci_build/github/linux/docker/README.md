# Refreshing ORT build images

The manylinux CPU and CUDA images extend separately maintained upstream base
images. Changing this directory does not rebuild or publish those upstream bases.

`tools/ci_build/get_docker_image.py` always requests a fresh base-image resolution,
including when it uses a registry-backed buildx layer cache. Ordinary builds
continue to reuse cached layers. For an authorized security refresh, include
`--no-cache` in `--docker-build-args` to rerun package installation even when the
base digest has not changed. The `DockerBuildArgs` pipeline-template parameter
forwards this option to the helper.

The manylinux CPU, CUDA and WebGPU images inherit Node/npm from their producer
bases instead of reinstalling them. Custom `BASEIMAGE` overrides must also provide
Node/npm.

Build and promote upstream test images before changing `BASEIMAGE` defaults or
pipeline/workflow overrides for validation. Use verified immutable digests, preserve the
consuming image's OS/manylinux ABI and CUDA compatibility, and keep exported
CUDA-plugin metadata consistent. Rebuild and scan the derived image too; moving
a registry tag does not replace a running container. Promotion is not release
approval: complete the validation and scan gates before merging consumer changes.

## Central base-image configuration

[`base-images.json`](base-images.json) is the source of truth for image
repositories, immutable digests, promotion tags and known producer build IDs.
Its `dockerfile_defaults` map selects a family for each managed Dockerfile.
For routine base refreshes, update only the affected `images` entries in this
manifest. Tags are provenance metadata; builds always use `repository@digest`.

ADO passes `BaseImageFamily` through `get-docker-image-steps.yml` to
`get_docker_image.py --base-image-family`. GitHub uses the local
`.github/actions/build-docker-image` action, optionally selecting a
`base-image-family` (or `base_image_family` in `reusable_linux_build.yml`).
Both call `resolve_docker_base_image.py`, using the Dockerfile's configured
default when no family or explicit `BASEIMAGE` argument is supplied.
Unknown families and conflicting family/explicit-image selections fail.
ADO and local helper calls continue to resolve ACR references. The GitHub wrapper
uses `--registry ghcr` and resolves managed families to
`ghcr.io/microsoft/onnxruntime/base/<family>@<inventory-digest>`. Both registries use
the same inventory pins; no second digest inventory is maintained.

The resolver CLI writes a single line to stdout: with `--family` alone, an
immutable `repository@sha256:digest` reference; with `--dockerfile`, shell-quoted
Docker build arguments, retaining supplied arguments and adding the configured
`BASEIMAGE` if needed. The GitHub wrapper captures that line and appends
`build-args=<resolved arguments>` to `$GITHUB_OUTPUT`, exposing
`steps.resolve.outputs.build-args` to the next step. Resolution failures exit
before any step output is written.

The GitHub wrapper resolves the digest before invoking the existing Docker
action, so the digest enters its build-argument cache key. A manifest update
therefore invalidates the appropriate derived-image cache without changing the
existing registry publishing policy. CUDA plugin metadata also records the
resolved immutable reference so downstream tests use the packaging run's base,
not the current manifest's selection.

Managed Dockerfiles intentionally declare `ARG BASEIMAGE` without a pinned
default. From the repository root, use the helper for local builds:

```shell
python tools/ci_build/get_docker_image.py \
  --dockerfile tools/ci_build/github/linux/docker/Dockerfile.manylinux2_28_cuda \
  --context tools/ci_build/github/linux/docker \
  --repository ort-cuda-local \
  --base-image-family cuda13_x64_almalinux8_gcc14 \
  --docker-build-args="--build-arg TRT_VERSION="
```

Omit `--base-image-family` to use the manifest's Dockerfile default.
Without `--container-registry`, this helper does not push an image.
For a raw `docker build`, explicitly supply `--build-arg BASEIMAGE=<reference>`;
Docker itself cannot read the manifest. Explicit image arguments remain
supported for custom images and historical packaging metadata.

## Promoting approved bases from ACR to GHCR

ADO remains the trusted producer and stages images in ACR. The
[`Promote approved base images`](../../../../../.github/workflows/promote_base_images.yml)
workflow copies inventory-approved images without rebuilding them. Its manual path accepts
one `family` from `base-images.json`, or `all`; it does not accept registry
locations, tags, digests, scripts or checkout refs as inputs.

The mapping is:

| Purpose | Reference |
| --- | --- |
| Source | `<inventory repository>@<inventory digest>` |
| Promoted version | `ghcr.io/microsoft/onnxruntime/base/<family>:<inventory tag>` |
| GitHub build input | `ghcr.io/microsoft/onnxruntime/base/<family>@<inventory digest>` |

Skopeo copies directly between registries, without a Docker daemon.
[`skopeo copy --all --preserve-digests`](https://github.com/containers/skopeo/blob/v1.15.0/docs/skopeo-copy.1.md)
copies all platforms, the original manifest/index and its referenced layers,
and fails if digests cannot be preserved. A single-platform Docker
pull/tag/push is not equivalent. The helper hashes the raw source manifest
against the inventory pin, rejects conflicting destination version tags or
incompatible existing pinned content, then verifies both the destination tag
and digest against the exact source bytes. Reruns request the same full-platform
copy rather than skipping it based on the root digest. No manifest conversion is requested.
This does not copy unrelated OCI referrers or establish new scan/signing approval.

The workflow only runs in `microsoft/onnxruntime`, either on protected `main` for
`workflow_dispatch`, or on the exact protected pilot branch below for `push`.
Both paths require approval through the configured promotion environment and use
the triggering commit's reviewed code. The Python helper checks the same repository,
event, protected ref and workflow path before credentials are accessed. There is no
`pull_request_target`, fork checkout or automatic publishing from a consumer job.
Workflow concurrency serializes promotions across all families. Registry
tag checks are not an atomic compare-and-swap against independent publishers:
owners must reserve this destination namespace for this workflow and prevent
other writers from racing it. Promotion of `all` is not transactional; an error
stops further copies but does not roll back already verified images.

### Owner prerequisites and operation

Before enabling promotion or relying on anonymous fork consumption:

1. Approve redistribution of each producer image and its included SDKs/vendor
   payloads. An approved inventory pin is not evidence of licensing or public
   redistribution authorization, and promotion is not a clean-scan claim.
2. Configure the `base-image-promotion` GitHub environment with required
   reviewers and deployment branches restricted to `main` (and the exact pilot
   branch only during pre-merge testing); protect and require review of those
   branches. An environment name in YAML does not create these protections.
3. Configure an Azure OIDC reader identity for this environment's subject
   `repo:microsoft/onnxruntime:environment:base-image-promotion` and audience
   `api://AzureADTokenExchange`. Grant only the ACR/repository read permissions
   needed for the approved source namespace, not ACR write. The existing PR-reader
   federation does not automatically authorize this environment.
4. Supply public configuration variables `ACR_READ_CLIENT_ID`,
   `AZURE_TENANT_ID` and `ACR_PROMOTION_SUBSCRIPTION_ID` at repository or environment
   scope. Client/tenant names align with the ACR reader configuration. The pilot
   found that the existing generic `AZURE_SUBSCRIPTION_ID` points to a different
   subscription than `onnxruntimebuildcache`; do not reuse it or change it globally.
   Owners must independently verify the registry's subscription and configure the
   distinct `ACR_PROMOTION_SUBSCRIPTION_ID` for promotion.
   Variables are resolved at the workflow boundary, not inside composite actions.
5. Allow the workflow's `GITHUB_TOKEN` to publish these GHCR packages with
   `packages:write`. For pre-existing packages, grant this repository Actions
   access and check package ownership/association. Do not substitute a personal
   token or grant publisher credentials to fork jobs.

After those approvals and once the workflow is on `main`, an authorized upstream maintainer can select
**Actions > Promote approved base images > Run workflow**, choose **main** and
the desired **family** (or **all**). Equivalent CLI invocation:

```shell
gh workflow run promote_base_images.yml --repo microsoft/onnxruntime --ref main -f family=all
```

### Pre-merge integration test

GitHub requires a new `workflow_dispatch` workflow to exist on the default branch
before it can be manually dispatched. To test the actual promotion implementation
without merging to `main`, a temporary `push` trigger is restricted to this exact
upstream branch:

```text
eserscor-2026-11-12-base-image-promotion
```

An owner must first protect that branch, require review, and configure the
`base-image-promotion` environment's required reviewers and deployment branch
allowlist to include that exact branch. The environment-specific Azure OIDC trust,
ACR read permissions, scoped subscription configuration and redistribution approvals
above are still required. Branch/environment settings are not changed by this code.
Do not weaken the guards or supply credentials to a fork to get around these prerequisites.

After review, an authorized maintainer pushes the reviewed commit to that branch
in `microsoft/onnxruntime` and approves the environment deployment. This invokes the
same promotion job and helper as the production path, with `family=all`, so every
approved family needed by downstream CI is available. It is a real publication,
not a dry run. The same branch name in a fork, unprotected upstream branches,
other protected branches and PR events cannot promote.

Once an owner approves public visibility and verifies anonymous pulls, dispatch
the existing CI workflow against the test branch:

```shell
gh workflow run linux_minimal_build.yml --repo microsoft/onnxruntime --ref eserscor-2026-11-12-base-image-promotion
```

For fork validation, put the reviewed changes on a branch in a fork and open an
ordinary pull request against upstream `main`, without merging it. Existing
`pull_request` CI then tests the proposed GHCR consumer changes from the PR merge
ref. Confirm the GHCR preflight and Docker build/pull work without ACR login.
The promotion workflow must not run in that PR or gain publisher credentials.
This tests base-image distribution, not separate Windows SDK Blob authorization.

Rerun the same approved promotion run from Actions to test idempotence; an
authorized maintainer can also use `gh run rerun <promotion-run-id> --repo microsoft/onnxruntime`.
Do not delete packages or invent a new pin to exercise missing-image/collision
handling; use the offline mocks for those negative cases.

Remove the temporary push trigger, pilot job-guard alternative, Python
`PILOT_BRANCH` allowance, pilot-specific tests/error guidance and this section
before the final production merge. Owners should then remove the pilot branch
from the environment deployment allowlist. Production remains manual on protected
`main`; consumer PR jobs never publish.

New GHCR packages may be **private**, even for a public repository. Following
redistribution approval, a package owner must explicitly make each package public
and verify anonymous access to the pinned digest. The workflow never changes
visibility, Azure RBAC/federation, GitHub variables/settings or runner identities.
A successful authenticated promotion does **not** prove anonymous pull availability.

Registry credentials are written only to a private, temporary Skopeo auth file;
the GitHub token is removed from the child-process environment and credentials
are not passed on command lines or printed. The auth file is removed on success
and exceptions. Azure CLI uses an isolated cache under the hosted runner's
temporary directory, removed by an `always()` step, also covering interrupted
copy cleanup. Hard termination still relies on disposal of the hosted runner.
Command failures report the failed tool/subcommand and exit code; raw tool
output is withheld to avoid exposing credential-bearing diagnostics.

### GitHub consumer failures

Managed GitHub consumers default to GHCR. Before invoking the Docker build/cache
action, the wrapper anonymously requests the inventory-pinned manifest and
verifies its digest. It does not use publisher credentials. The wrapper does not
accept or pass an ACR registry name to the downstream action, so it never requests
the downstream action's managed-identity ACR login. Explicit custom/historical
ACR overrides require separately provisioned authentication; this build action
does not provide it.

A structured registry `404` with `MANIFEST_UNKNOWN` or `NAME_UNKNOWN` produces
an error identifying the missing digest, the promotion workflow, the selected
family and the command an authorized maintainer must run before retrying CI.
There is no ACR fallback and no automatic workflow dispatch. Authentication or
visibility denials (`401`/`403`), rate limits (`429`), network/TLS errors and
unrecognized `404`s fail separately: they do not prove an image is absent and
do not recommend publication. GHCR can deny anonymous token acquisition for a
private or nonexistent package, so such a denial cannot safely be classified
as a missing image.

Until owners finish promotion **and** public visibility verification, this
default-GHCR rollout can block GitHub builds. Existing ADO/local ACR behavior
and separate Windows SDK Blob access are unchanged. Offline unit/mock tests
cover selection, digest checks, copy flags, collisions, errors, cleanup and
trust guards; they do not establish production promotion or public-pull success.

## Current official base images

The manifest records the nine consumed official AlmaLinux8 and AlmaLinux9 families alongside
their tags, digests and producer build IDs. Its `producer_builds` entries record
the source commits for those builds. ADO and GitHub consumers
select these families rather than duplicating their pins or current release
provenance in this document or CI files.

CUDA13 ARM64 selections use the refreshed AlmaLinux9 family from the same
official producer build as the AlmaLinux8 families. Its repository now uses the
shared promotion namespace rather than the historical image's namespace.
The AlmaLinux8 ARM64 CUDA family is not an ABI-compatible replacement.
The new bases provide CPython 3.11--3.14, not 3.10, so CUDA, TensorRT and WebGPU
CI jobs that previously selected `/opt/python/cp310-cp310` now select
`/opt/python/cp311-cp311` consistently for build and test.

The dependency refresh alone is not a clean-scan claim. npm 11.21.1 still bundles
`ip-address` 10.7.2, below the report's 10.7.4 target. CUDA's Nsight `nic_sampler`
also needs a patched NVIDIA payload, not a newer host Go compiler. Keep profiling
enabled and retain that vendor fix as an outstanding remediation gate.
