#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from unittest.mock import MagicMock

sys.modules["pycuda"] = MagicMock()
sys.modules["pycuda.compiler"] = MagicMock()
sys.modules["pycuda.cumath"] = MagicMock()
sys.modules["pycuda.driver"] = MagicMock()
sys.modules["pycuda.characterize"] = MagicMock()
sys.modules["pycuda.reduction"] = MagicMock()
sys.modules["pycuda._mymako"] = MagicMock()
sys.modules["pycuda.tools"] = MagicMock()
sys.modules["pycuda.elementwise"] = MagicMock()
sys.modules["pycuda._cluda"] = MagicMock()
sys.modules["pycuda.autoinit"] = MagicMock()
sys.modules["pycuda.scan"] = MagicMock()
sys.modules["pycuda.gpuarray"] = MagicMock()
sys.modules["pycuda.debug"] = MagicMock()
sys.modules["pycuda.__init__"] = MagicMock()
sys.modules["pycuda.curandom"] = MagicMock()
sys.modules["pycuda.compyte"] = MagicMock()
sys.modules["pycuda.sparse"] = MagicMock()
sys.modules["pycuda.cuda"] = MagicMock()
sys.modules["pycuda.gl"] = MagicMock()
