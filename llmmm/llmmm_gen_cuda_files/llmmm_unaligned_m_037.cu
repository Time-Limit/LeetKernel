#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_37_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<37> mm_instantiator;

public:
  UnalignedM_37_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_37_MMInstantiatorWrapper__;

}  // namespace LLMMM
