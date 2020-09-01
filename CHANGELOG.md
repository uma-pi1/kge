#### September 2020
- PR [#144](https://github.com/uma-pi1/kge/pull/144): Support to tune subbatch size automatically
- PR [#143](https://github.com/uma-pi1/kge/pull/143): Allow processing of large batches in smaller subbatches to save memory

#### August 2020
- PR [#140](https://github.com/uma-pi1/kge/pull/140): Calculate penalty for entities only once, if subject-embedder == object-embedder
- PR [#138](https://github.com/uma-pi1/kge/pull/138): Revision of hooks, fix of embedding normalization
- PR [#135](https://github.com/uma-pi1/kge/pull/135): Revised sampling API, faster negative sampling with shared samples

#### June 2020

- PR [#112](https://github.com/uma-pi1/kge/pull/112): Initialize embeddings from a packaged model
- PR [#113](https://github.com/uma-pi1/kge/pull/113): Reduce memory consumption and loading times of large datasets
- Various smaller improvements and bug fixes

#### May 2020

- PR [#110](https://github.com/uma-pi1/kge/pull/110): Support for different tie-breaking methods in evaluation (thanks Nzteb)
- [1d26e63](https://github.com/uma-pi1/kge/commit/1d26e63b65380e2c13db2ecb3986e69f404efdc2): Add head/tail evaluation per relation type 
- [dfd0aac](https://github.com/uma-pi1/kge/commit/dfd0aace1a77d6b7f04f414bdc8ea748a9d0d2f2): Added squared error loss (thanks Nzteb)
- PR [#104](https://github.com/uma-pi1/kge/pull/104): Fix incorrect relation type measurement (thanks STayinloves)
- PR [#101](https://github.com/uma-pi1/kge/pull/101): Revise embedder penalty API (thanks Nzteb)
- PR [#94](https://github.com/uma-pi1/kge/pull/94): Support for packaged models (thanks AdrianKS)
- Improved seeding of workers when a fixed NumPy seed is used
- Various smaller improvements and bug fixes
- Added more mappings from entity IDs to names for Freebase datasets (in entity_strings.del file)

#### Apr 2020

- Improved shared negative sampling (WOR sampling, exclude positive triples from negative sample)
- PR [#86](https://github.com/uma-pi1/kge/pull/86): Support (s,?,o) queries for KvsAll training (thanks vonVogelstein)

#### Mar 2020

- [cf64dd2](https://github.com/uma-pi1/kge/commit/cf64dd2fcc4c5961bda2d9142ea1b08d41c16ba2): Fast dataset/index loading via cached pickle files
- [4bc86b1](https://github.com/uma-pi1/kge/commit/4bc86b18e5cfe0a4a596dd25fbdc8dde59dcafe9): Add support for chunking a batch when training with negative sampling
- [14dc926](https://github.com/uma-pi1/kge/commit/14dc9268b2e24f7db36dc95ae47e5e975016955b): Add ability to dump configs in various ways
- PR [#64](https://github.com/uma-pi1/kge/pull/64): Initial support for frequency-based negative sampling (thanks AdrianKS)
- PR [#77](https://github.com/uma-pi1/kge/pull/77): Simpler use of command-line interface (thanks cthoyt)
- [76a0077](https://github.com/uma-pi1/kge/commit/76a007731d98e00331f2f6ccb90b91cc8cf265dd): Added RotatE
- [7235e99](https://github.com/uma-pi1/kge/commit/7235e99784e056b6d0e162ce84f0c5e1eb410895): Added option to add a constant offset before computing BCE loss
- [67de6c5](https://github.com/uma-pi1/kge/commit/67de6c5c422c2adcefcc56f7738e04d0893c51ba): Added CP
- [a5ee441](https://github.com/uma-pi1/kge/commit/a5ee4417b92559b3624e3f737939793da810c211): Added SimplE

#### Feb 2020
- PR [#71](https://github.com/uma-pi1/kge/pull/71): Faster and more memory-efficient training with negative sampling (thanks AdrianKS)
- Initial release
