# Third-party notices

This repository contains or derives from the following third-party code. The submodules and Python
dependencies keep their own licences; they are not redistributed here.

## Vendored or derived code

| Where | From | Licence |
|---|---|---|
| `scripts/groot-libero-x/server/msgpack_numpy.py` | [openpi-client](https://github.com/Physical-Intelligence/openpi) (`openpi_client/msgpack_numpy.py`), copied verbatim; itself adapted from [msgpack-numpy](https://github.com/lebedov/msgpack-numpy) | Apache-2.0 (openpi, Physical Intelligence); BSD-3-Clause (msgpack-numpy, below) |
| `scripts/groot-libero-x/server/feature_policy.py` | Re-implements the `Gr00tPolicy._get_action` pipeline of NVIDIA [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) so it can also return the backbone hidden states; modified | Apache-2.0 (NVIDIA) |

The Apache-2.0 text is in [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt).

### msgpack-numpy (BSD-3-Clause)

Copyright (c) 2013-2022, Lev E. Givon. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following
  disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of Lev E. Givon nor the names of any contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Used as submodules or dependencies (not redistributed)

- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (submodule, Apache-2.0 for the code; model weights have their own NVIDIA terms).
- [LIBERO-X](https://github.com/meituan/LIBERO-X) (submodule, MIT for the code; the dataset on the Hub is CC BY 4.0).
  `scripts/groot-libero-x/tasks/` lists LIBERO-X task files and `selection.csv` carries metadata about them.
- [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0), [openpi-client](https://github.com/Physical-Intelligence/openpi) (Apache-2.0).
- Model weights: the pi0.5 base and adapter fall under the Gemma terms of use; GR00T's backbone is
  Cosmos-Reason2-2B under NVIDIA's licence. Check these before redistributing derived weights or features.
