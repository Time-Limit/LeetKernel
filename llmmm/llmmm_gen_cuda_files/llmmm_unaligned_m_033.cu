#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_33_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<33> mm_instantiator;

public:
  UnalignedM_33_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_33_MMInstantiatorWrapper__;

}  // namespace LLMMM
