# Copyright 2021 c00k1ez (https://github.com/c00k1ez). All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import youtokentome as yttm


class BPEWrapper(yttm.BPE):
    def __init__(
        self,
        pad_id: Optional[int] = 0,
        unk_id: Optional[int] = 1,
        bos_id: Optional[int] = 2,
        eos_id: Optional[int] = 3,
        *args,
        **kwargs,
    ) -> None:
        super(BPEWrapper, self).__init__(*args, **kwargs)
        assert self.vocab()[pad_id] == "<PAD>"
        assert self.vocab()[unk_id] == "<UNK>"
        assert self.vocab()[bos_id] == "<BOS>"
        assert self.vocab()[eos_id] == "<EOS>"
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id
