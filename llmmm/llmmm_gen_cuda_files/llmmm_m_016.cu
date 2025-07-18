#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_16_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<16> mm_instantiator;

public:
  UnalignedM_16_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_16_MMInstantiatorWrapper__;

}  // namespace LLMMM
