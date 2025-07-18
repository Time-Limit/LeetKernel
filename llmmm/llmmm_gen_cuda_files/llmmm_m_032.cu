#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_32_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<32> mm_instantiator;

public:
  UnalignedM_32_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_32_MMInstantiatorWrapper__;

}  // namespace LLMMM
