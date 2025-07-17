#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_34_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<34> mm_instantiator;

public:
  UnalignedM_34_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_34_MMInstantiatorWrapper__;

}  // namespace LLMMM
