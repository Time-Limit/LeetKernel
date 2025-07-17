#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_52_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<52> mm_instantiator;

public:
  UnalignedM_52_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_52_MMInstantiatorWrapper__;

}  // namespace LLMMM
