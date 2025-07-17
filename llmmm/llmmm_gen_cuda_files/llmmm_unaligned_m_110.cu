#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_110_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<110> mm_instantiator;

public:
  UnalignedM_110_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_110_MMInstantiatorWrapper__;

}  // namespace LLMMM
