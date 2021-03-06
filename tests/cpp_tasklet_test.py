#!/usr/bin/env python
import dace


@dace.program(dace.uint32[2], dace.uint32[1])
def cpp_tasklet(A, B):
    @dace.tasklet('CPP', global_code='#include <cstdio>')
    def index2():
        a << A[0]
        b >> B[0]
        """
        b = a;
        printf("I have been added as raw C++ code\\n");
        """


if __name__ == '__main__':

    A = dace.ndarray((2, ), dace.uint32)
    B = dace.ndarray((1, ), dace.uint32)

    A[:] = 5
    B[:] = 0

    cpp_tasklet(A, B)

    if B[0] != 5:
        raise RuntimeError("Expected output {}, got {}".format(5, B))
