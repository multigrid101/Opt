terra Index.initFromCUDAParams(self : &Index) : bool
    (@self).d0, (@self).d1 = llvm.nvvm.read.ptx.sreg.ntid.x() * llvm.nvvm.read.ptx.sreg.ctaid.x() + llvm.nvvm.read.ptx.sreg.tid.x(), llvm.nvvm.read.ptx.sreg.ntid.y() * llvm.nvvm.read.ptx.sreg.ctaid.y() + llvm.nvvm.read.ptx.sreg.tid.y()
    return true and (@self).d0 < 1000 and (@self).d1 < 2000
end

