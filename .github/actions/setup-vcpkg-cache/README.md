# vcpkg binary cache

Call this action after selecting the build toolchain. Windows callers must first
initialize the MSVC environment (normally with `locate-vcvarsall-and-setup-env`).
The selected compiler must agree with the compiler used by CMake and vcpkg; do not
switch toolsets after restoring the cache.

Windows cache keys include the selected `cl.exe` content hash, `VCToolsVersion`,
Windows SDK version, and host/target architectures, in addition to the existing
runner, job/build-variant, vcpkg release, manifest, overlay, and triplet-generator
identities. The compiler hash is necessary because an MSVC servicing update can
change the compiler binary while leaving the toolset directory/version unchanged.
Both exact keys and restore prefixes are toolchain-scoped so a new compiler can
populate a new immutable archive. Non-Windows keys are unchanged.

Only vcpkg binary archives are cached. vcpkg still computes and checks each
package's ABI, including compiler tracking and generated triplet contents; this
action does not disable those checks or change compiler/security flags. A GitHub
cache hit only means that an archive was downloaded. Check vcpkg's
`Restored N package(s)` and `Building ...` output to establish actual binary reuse.
Only main builds publish archives; pull requests restore without saving.

Focused validation:

```powershell
python -m unittest discover -s tools\ci_build -p test_vcpkg_cache.py
```

To also compile a small native package with the generated BinSkim triplet and
verify binary reuse from copied archives in a fresh install/cache directory,
run the same command in an x64 MSVC developer environment with
`ORT_VCPKG_CACHE_TEST_EXE` set to the pinned `vcpkg.exe`. The smoke test uses only
temporary directories; it does not touch the user's binary cache. It verifies
same-toolchain reuse, while unit tests verify separation after compiler-content,
SDK, architecture, and variant changes.

Set `ORT_ACTIONLINT_EXE` to `actionlint` to also validate the composite steps in a
typed workflow wrapper.
