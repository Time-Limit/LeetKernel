#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_84_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<84> mm_instantiator;

public:
  UnalignedM_84_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_84_MMInstantiatorWrapper__;

}  // namespace LLMMM
