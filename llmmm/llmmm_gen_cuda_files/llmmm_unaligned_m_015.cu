#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_15_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<15> mm_instantiator;

public:
  UnalignedM_15_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_15_MMInstantiatorWrapper__;

}  // namespace LLMMM
