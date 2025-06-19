---
title: Versioning Guide
---

# Versioning Guide


The Outlines project follows a structured versioning scheme designed to provide clarity and minimize risk for downstream dependents.

Each part of the version number (`major.minor.patch`) conveys information about the nature and impact of the changes included in the release.

- **Major Releases** includes compatibility-breaking changes to core interfaces, such as `LogitsProcessor`s and `Guides`.
- **Minor Releases** introduce changes of substance to internal or unexposed functionality. These changes are well tested and intended to maintain compatibility with existing use of core interfaces.
- **Patch Releases** address bug fixes and incorporate low-risk changes to improve stability and performance.

!!! note "Breaking Changes"

    Outlines v1.0 introduced several breaking changes to the core interface. See [the migration guide](/user_guide/migration) for more details.

## Releases

Releases along with release notes can be found on the [Outlines Releases GitHub Page](https://github.com/dottxt-ai/outlines/releases).

## Version Pinning Recommendations

Here are our recommendations for managing dependencies on the Outlines package:

**Small, Risk-Tolerant Projects:** Pin to a specific major version.

**Large, Conservative Projects:** Pin to a specific minor version.
