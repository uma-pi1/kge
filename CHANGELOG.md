#### October 2021
- [5127cf2](https://github.com/uma-pi1/kge/commit/5127cf287488a3abfc2e8b698fad22085829a9aa): Fix differences in TransE scoring implementations

#### September 2021
- PR [#224](https://github.com/uma-pi1/kge/pull/224): Take floating point issues into account for tie calculation in entity-ranking (thanks @sfschouten)

#### June 2021
- [9a4f69a](https://github.com/uma-pi1/kge/commit/9a4f69a806ea775fed13657a3e3cd1c6711a2279): Refactor  time measurement with new timer class

#### March 2021
- PR [#191](https://github.com/uma-pi1/kge/pull/191): Fix loading of pretrained embeddings with reciprocal relation models

#### February 2021
- [27e8a32](https://github.com/uma-pi1/kge/commit/27e8a323d208106d7b75f4e003ea4b73c1c5d58d): improve validation time by allowing bulk KvsAll index lookup and improved history computation
- PR [#154](https://github.com/uma-pi1/kge/pull/154): store checkpoint containing the initialized model for reproducibility
- [9e88117](https://github.com/uma-pi1/kge/commit/9e88117b3bf3f91b1c22f17d88eae2f77b5e3d3e): Add Transformer model and learning rate warmup (thanks nluedema)
- PR [#176](https://github.com/uma-pi1/kge/pull/176): Add TransH model (thanks Mayo42)

#### December 2020
- PR [#164](https://github.com/uma-pi1/kge/pull/164): Allow to easily add custom training/evaluation/search jobs
- PR [#159](https://github.com/uma-pi1/kge/pull/159): Add a plugin mechanism (thanks @sfschouten)
- PR [#157](https://github.com/uma-pi1/kge/pull/157): Add CoDEx datasets and pretrained models (thanks @tsafavi)

#### November 2020
- PR [#155](https://github.com/uma-pi1/kge/pull/155): Faster reading of triple files

#### October 2020
- [d275419](https://github.com/uma-pi1/kge/commit/d275419bbd1e2eea6872d733fc10f30c171e9f45), [87c5463](https://github.com/uma-pi1/kge/commit/87c54630807e7ecf71ad05c042d3b1c953c44807): Support parameter groups with group-specific optimizer args
- PR [#152](https://github.com/uma-pi1/kge/pull/152): Added training loss evaluation job

#### September 2020
- PR [#147](https://github.com/uma-pi1/kge/pull/147): Support both minimization and maximization of metrics
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
