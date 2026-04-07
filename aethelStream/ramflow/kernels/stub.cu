// kernels/stub.cu — Sprint 0 linkable stub (one per build)
//
// This file is compiled first in build.rs and archived into
// libramflow_cuda_stub.a.  Once Sprint 2 is implemented, overflow_check.o
// and overflow_density.o will be archived alongside it.

extern "C" {
    void ramflow_cuda_stub_init(void);
}

void ramflow_cuda_stub_init(void) {}
