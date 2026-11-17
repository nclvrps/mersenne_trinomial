// minimal NTL link check used by the Makefile's auto-detection
#include <NTL/GF2X.h>
int main() {
    NTL::GF2X a;
    NTL::SetCoeff(a, 3);
    NTL::SetCoeff(a, 0);
    return NTL::deg(a) == 3 ? 0 : 1;
}
