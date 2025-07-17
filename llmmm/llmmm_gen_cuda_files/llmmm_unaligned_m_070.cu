#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_70_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<70> mm_instantiator;

public:
  UnalignedM_70_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_70_MMInstantiatorWrapper__;

}  // namespace LLMMM
