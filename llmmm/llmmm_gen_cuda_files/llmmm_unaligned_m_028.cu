#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_28_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<28> mm_instantiator;

public:
  UnalignedM_28_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_28_MMInstantiatorWrapper__;

}  // namespace LLMMM
