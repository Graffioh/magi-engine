#include "test_utils.h"

#include <iostream>

void run_tensor_tests(TestState&);
void run_ops_tests(TestState&);
void run_mmap_tests(TestState&);

int main() {
    TestState s;

    std::cout << "=== tensor ===\n";
    run_tensor_tests(s);

    std::cout << "\n=== ops ===\n";
    run_ops_tests(s);

    std::cout << "\n=== mmap ===\n";
    run_mmap_tests(s);

    std::cout << "\nresults: " << s.passed << " passed, " << s.failed << " failed\n";
    return s.failed == 0 ? 0 : 1;
}
