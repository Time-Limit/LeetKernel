#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_78_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<78> mm_instantiator;

public:
  UnalignedM_78_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_78_MMInstantiatorWrapper__;

}  // namespace LLMMM
