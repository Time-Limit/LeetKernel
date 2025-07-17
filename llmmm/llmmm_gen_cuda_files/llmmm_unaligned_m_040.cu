#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_40_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<40> mm_instantiator;

public:
  UnalignedM_40_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_40_MMInstantiatorWrapper__;

}  // namespace LLMMM
